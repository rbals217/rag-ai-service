import re
import uuid
from dataclasses import dataclass

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings import Settings


# ───────────────────────── 청크 데이터 ─────────────────────────

@dataclass
class Chunk:
    concert_id: int
    title: str
    field_name: str
    text: str
    score: float = 0.0


# ───────────────────────── 공통 도우미 ─────────────────────────

def make_splitter(settings: Settings) -> RecursiveCharacterTextSplitter:
    # 자연스러운 문단과 문장 경계를 우선하여 청크 분할
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_texts(
    splitter: RecursiveCharacterTextSplitter,
    title: str,
    field_name: str,
    content: object
) -> list[str]:
    """공연의 특정 항목 하나를 독립적으로 분할한다."""

    if content is None:
        return []

    content = str(content).strip()

    # 값이 비어 있는 항목은 청크를 만들지 않는다.
    if not content:
        return []

    pieces = splitter.split_text(content)

    return [
        (
            f"공연 제목: {title}\n"
            f"정보 항목: {field_name}\n"
            f"{piece}"
        ).strip()
        for piece in pieces
    ]


def make_concert_chunks(
    splitter: RecursiveCharacterTextSplitter,
    title,
    price,
    location,
    notice,
    casting,
    detail,
    price_info,
    guide,
    refund,
    story,
    mood
) -> list[tuple[str, str]]:
    """11개 공연 항목을 각각 독립적으로 청크화한다."""

    fields = {
        "공연 제목": title,
        "가격": price,
        "장소": location,
        "공지사항": notice,
        "출연진": casting,
        "공연 상세정보": detail,
        "가격 정보": price_info,
        "관람 안내": guide,
        "환불 정보": refund,
        "공연 스토리": story,
        "공연 분위기": mood,
    }

    all_chunks = []

    for field_name, content in fields.items():
        field_chunks = chunk_texts(
            splitter,
            title,
            field_name,
            content
        )

        # 항목명과 청크 본문을 함께 보관한다.
        all_chunks.extend(
            (field_name, chunk_text)
            for chunk_text in field_chunks
        )

    return all_chunks


def body_of(text: str) -> str:
    """청크에서 공연 제목과 정보 항목 표시를 제외한 본문을 반환한다."""

    lines = text.split("\n")

    body_lines = [
        line for line in lines
        if not line.startswith("공연 제목:")
        and not line.startswith("정보 항목:")
    ]

    return "\n".join(body_lines).strip()


def format_context(chunks: list[Chunk]) -> str:
    """검색된 공연 청크를 LLM에 전달할 근거 목록으로 변환한다."""

    return "\n\n".join(
        f"[#공연ID:{chunk.concert_id}] "
        f"공연 제목: {chunk.title}\n"
        f"정보 항목: {chunk.field_name}\n"
        f"{body_of(chunk.text)}"
        for chunk in chunks
    )


def bigrams(text: str) -> set[str]:
    """문자열을 두 글자 단위로 나눈다."""

    cleaned = re.sub(r"[^0-9a-zA-Z가-힣]", "", text.lower())

    if len(cleaned) > 1:
        return {
            cleaned[i:i + 2]
            for i in range(len(cleaned) - 1)
        }

    return {cleaned} if cleaned else set()


# ───────────────────────── 질문 프롬프트 ─────────────────────────

_STOP = [
    "하나요", "인가요", "있나요", "되나요", "할까요",
    "어때", "알려줘", "알려 줘", "궁금해","악기","커플",""
    "뭐야", "언제", "어디", "어떻게", "무엇", "?", "？"
]


ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "너는 공연 정보를 안내하는 AI 도우미다. "
        "아래 [공연 정보]에 있는 내용만 근거로 한국어로 답한다.\n"
        "- 공연 정보에 없는 내용은 추측하지 않는다.\n"
        "- 찾을 수 없는 정보는 '해당 공연 정보에서 찾을 수 없습니다'라고 답한다.\n"
        "- 근거로 사용한 문장 끝에는 [#공연ID:번호]를 붙인다.\n"

    ),
    (
        "human",
        "[공연 정보]\n{context}\n\n[질문]\n{question}"
    ),
])


REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "사용자 질문을 공연 정보 검색에 적합한 짧은 검색 문장으로 바꾼다. "
        "공연명, 가격, 장소, 출연진 등 핵심 명사와 키워드를 남기고 "
        "한 줄로만 출력한다."
    ),
    ("human", "{question}"),
])


ABSTAIN_MESSAGE = (
    "관련 공연 정보를 찾지 못했습니다. "
    "공연명이나 질문을 조금 더 구체적으로 입력해 주세요."
)


# ───────────────────────── Stub 엔진 ─────────────────────────

class StubEngine:
    """메모리에 공연 청크를 저장하고 2-gram으로 검색하는 엔진."""

    name = "stub"

    def __init__(self, settings: Settings):
        self.splitter = make_splitter(settings)
        self.min_score = settings.stub_min_score

        # 공연 ID별 원본 공연 정보
        self.concert_data = {}

        # 공연 ID별 청크 목록
        self.store: dict[int, list[Chunk]] = {}

    # 공연 색인
    def index_concert(
        self,
        concert_id,
        title,
        price,
        location,
        notice,
        casting,
        detail,
        price_info,
        guide,
        refund,
        story,
        mood
    ):
        # 공연 원본 정보 저장
        self.concert_data[concert_id] = {
            "type": "CONCERT",
            "concert_id": concert_id,
            "title": title,
            "price": price,
            "location": location,
            "notice": notice,
            "casting": casting,
            "detail": detail,
            "price_info": price_info,
            "guide": guide,
            "refund": refund,
            "story": story,
            "mood": mood,
        }

        # 11개 항목을 각각 독립적으로 청크화
        field_chunks = make_concert_chunks(
            self.splitter,
            title,
            price,
            location,
            notice,
            casting,
            detail,
            price_info,
            guide,
            refund,
            story,
            mood
        )

        # 공연 ID별 청크 저장
        self.store[concert_id] = [
            Chunk(
                concert_id=concert_id,
                title=title,
                field_name=field_name,
                text=text
            )
            for field_name, text in field_chunks
        ]

        # 청크 분할 결과 출력
        for i, chunk in enumerate(self.store[concert_id]):
            print(f"공연 ID: {concert_id}")
            print(f"청크 번호: {i + 1}")
            print(f"정보 항목: {chunk.field_name}")
            print(f"청크 길이: {len(chunk.text)}")
            print(f"청크 내용: {chunk.text}")
            print("-" * 50)

        return len(field_chunks)

    # 공연 삭제
    def delete_concert(self, concert_id):
        existed = concert_id in self.store

        self.store.pop(concert_id, None)
        self.concert_data.pop(concert_id, None)

        return 1 if existed else 0

    # 공연 검색
    def search(self, query, concert_ids, k):

        # 검색 허용 공연 ID가 없으면 검색하지 않는다.
        if not concert_ids:
            return []

        query_bigrams = bigrams(query)
        allowed = set(concert_ids)
        scored = []

        for concert_id, chunks in self.store.items():

            # 허용된 공연 ID가 아니면 제외한다.
            if concert_id not in allowed:
                continue

            for chunk in chunks:
                chunk_bigrams = bigrams(chunk.text)

                if not query_bigrams or not chunk_bigrams:
                    continue

                score = (
                    len(query_bigrams & chunk_bigrams)
                    / len(query_bigrams | chunk_bigrams)
                )

                if score < self.min_score:
                    continue

                scored.append(
                    Chunk(
                        concert_id=chunk.concert_id,
                        title=chunk.title,
                        field_name=chunk.field_name,
                        text=chunk.text,
                        score=round(score, 3)
                    )
                )

        scored.sort(
            key=lambda chunk: chunk.score,
            reverse=True
        )

        return scored[:k]

    # 검색 결과를 이용한 답변
    def generate(self, question, chunks):
        if not chunks:
            return ABSTAIN_MESSAGE

        best = chunks[0]
        body = body_of(best.text).replace("\n", " ")[:150]

        return (
            f"'{best.title}' 공연의 {best.field_name} 정보에 따르면: "
            f"{body} [#공연ID:{best.concert_id}]"
        )

    # 질문 재작성
    def rewrite(self, question):
        query = question

        for word in _STOP:
            query = query.replace(word, " ")

        return " ".join(query.split()) or question


# ───────────────────────── Atlas 엔진 ─────────────────────────

class AtlasEngine:
    """OpenAI 임베딩과 MongoDB Atlas Vector Search를 사용하는 엔진."""

    name = "atlas"

    def __init__(self, settings: Settings):
        from langchain_mongodb import MongoDBAtlasVectorSearch
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from pymongo import MongoClient
        import certifi

        if not settings.openai_api_key or not settings.mongodb_uri:
            raise RuntimeError(
                "RAG_ENGINE=atlas에는 "
                "OPENAI_API_KEY와 MONGODB_URI가 필요합니다."
            )

        self.splitter = make_splitter(settings)
        self.min_score = settings.min_score
        self.settings = settings

        # MongoDB 연결
        self.client = MongoClient(
            settings.mongodb_uri,
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=10000
        )

        # Settings에 정의된 실제 변수명 사용
        self.collection = self.client[
            settings.mongodb_db
        ][
            settings.mongodb_collection
        ]

        # OpenAI 임베딩 객체 생성
        embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            dimensions=settings.embedding_dimensions,
            api_key=settings.openai_api_key
        )

        # MongoDB Atlas Vector Search 연결
        self.store = MongoDBAtlasVectorSearch(
            collection=self.collection,
            embedding=embeddings,
            index_name=settings.atlas_index_name,
            text_key="text",
            embedding_key="embedding",
            relevance_score_fn="cosine"
        )

        # ChatOpenAI 모델 생성
        llm = ChatOpenAI(
            model=settings.openai_chat_model,
            temperature=0.6,
            api_key=settings.openai_api_key
        )

        # 질문과 검색 자료를 이용한 답변 생성 체인
        self.answer_chain = (
            ANSWER_PROMPT
            | llm
            | StrOutputParser()
        )

        # 사용자 질문 재작성 체인
        self.rewrite_chain = (
            REWRITE_PROMPT
            | llm
            | StrOutputParser()
        )

    # 공연 색인
    def index_concert(
        self,
        concert_id,
        title,
        price,
        location,
        notice,
        casting,
        detail,
        price_info,
        guide,
        refund,
        story,
        mood
    ):
        # 11개 항목을 각각 독립적으로 청크화
        field_chunks = make_concert_chunks(
            self.splitter,
            title,
            price,
            location,
            notice,
            casting,
            detail,
            price_info,
            guide,
            refund,
            story,
            mood
        )

        # 청크가 없다면 기존 색인 데이터를 삭제하지 않는다.
        if not field_chunks:
            return 0

        # 재색인 버전을 구분하기 위한 고유 ID 생성
        index_version = str(uuid.uuid4())

        texts = [
            chunk_text
            for _, chunk_text in field_chunks
        ]

        # 각 청크에 공연 ID, 제목, 항목 이름을 메타데이터로 저장
        metadatas = [
            {
                "type": "CONCERT",
                "concert_id": concert_id,
                "title": title,
                "field_name": field_name,
                "_index_version": index_version,
            }
            for field_name, _ in field_chunks
        ]

        # 새 청크와 임베딩을 먼저 저장한다.
        self.store.add_texts(
            texts=texts,
            metadatas=metadatas
        )

        # 새 청크 저장이 성공한 다음 기존 청크를 삭제한다.
        self.collection.delete_many({
            "concert_id": concert_id,
            "_index_version": {"$ne": index_version}
        })

        return len(texts)

    # 공연 삭제
    def delete_concert(self, concert_id):
        result = self.collection.delete_many({
            "concert_id": concert_id
        })

        return result.deleted_count

    # 공연 검색
    def search(self, query, concert_ids, k):

        # 검색 허용 공연 ID가 없으면 검색하지 않는다.
        if not concert_ids:
            return []

        # 우선 유사도가 높은 청크를 넉넉하게 검색한다.
        documents = self.store.similarity_search_with_score(
            query,
            k=max(k * 10, 50)
        )

        print("Atlas 검색 결과 개수:", len(documents))

        allowed = set(concert_ids)

        print("검색 허용 공연 ID:", concert_ids)
        print("검색 후 허용 ID:", allowed)

        scored = []

        for document, score in documents:
            metadata = document.metadata or {}
            concert_id = metadata.get("concert_id")

            # 허용된 공연 ID에 해당하지 않으면 제외한다.
            if concert_id not in allowed:
                continue

            score = float(score)

            print(
                "검색된 공연 ID:", concert_id,
                "| 점수:", score,
                "| 공연명:", metadata.get("title")
            )

            if score < self.min_score:
                continue

            scored.append(
                Chunk(
                    concert_id=concert_id,
                    title=metadata.get("title", ""),
                    field_name=metadata.get("field_name", ""),
                    text=document.page_content,
                    score=round(score, 3)
                )
            )

        scored.sort(
            key=lambda chunk: chunk.score,
            reverse=True
        )

        return scored[:k]

    # 검색 결과를 이용한 답변 생성
    def generate(self, question, chunks):
        if not chunks:
            return ABSTAIN_MESSAGE

        context = format_context(chunks)

        return self.answer_chain.invoke({
            "context": context,
            "question": question
        })

    # 질문 재작성
    def rewrite(self, question):
        return self.rewrite_chain.invoke({
            "question": question
        })

    # MongoDB 연결 종료
    def close(self):
        self.client.close()

    # Vector Search 인덱스 생성
    def create_vector_index(self):
        from pymongo.operations import SearchIndexModel

        index_name = self.settings.atlas_index_name

        # 기존 인덱스 확인
        indexes = list(self.collection.list_search_indexes())

        for index in indexes:
            if index.get("name") == index_name:
                return {
                    "index_name": index_name,
                    "created": False,
                    "message": "이미 존재하는 인덱스입니다."
                }

        # Vector Search 인덱스 정의
        index_model = SearchIndexModel(
            definition={
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": self.settings.embedding_dimensions,
                        "similarity": "cosine"
                    }
                ]
            },
            name=index_name,
            type="vectorSearch"
        )

        # MongoDB Atlas에 인덱스 생성 요청
        created_name = self.collection.create_search_index(
            model=index_model
        )

        return {
            "index_name": created_name,
            "created": True,
            "message": "Vector Search 인덱스 생성 요청이 완료되었습니다."
        }

    # Vector Search 인덱스 상태 확인
    def get_vector_index_status(self):
        index_name = self.settings.atlas_index_name

        indexes = list(self.collection.list_search_indexes())

        for index in indexes:
            if index.get("name") == index_name:
                return {
                    "index_name": index_name,
                    "exists": True,
                    "status": index.get("status", "UNKNOWN"),
                    "queryable": index.get("queryable", False)
                }

        return {
            "index_name": index_name,
            "exists": False,
            "status": "NOT_FOUND",
            "queryable": False
        }


# ───────────────────────── 엔진 생성 ─────────────────────────

def create_engine(settings: Settings):
    if settings.rag_engine == "stub":
        return StubEngine(settings)

    if settings.rag_engine == "atlas":
        return AtlasEngine(settings)

    raise ValueError(
        f"지원하지 않는 RAG_ENGINE입니다: {settings.rag_engine}"
    )