
# 임시 RAG 엔진
# MongoDB Atlas 연결 전까지 공연 데이터를 메모리에 저장

class StubEngine:
    name = "stub"
    min_score = 0.0

    def __init__(self, settings):
        self.store = {}

        # 청크 분할 설정
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap

    # 청크 분할
    def split_text(self, text):
        chunks = []

        text = text.strip()

        if not text:
            return chunks

        start = 0

        while start < len(text):
            end = start + self.chunk_size

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            start += self.chunk_size -self.chunk_overlap

        return chunks

    # 공연 색인
    def index_concert(self, concert_id, title, price, location,notice,casting,detail,price_info,guide,refund,story,mood):
        self.store[concert_id] = {
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
            "story":story,
            "mood": mood,

        }
        # ② 공연 정보를 하나의 텍스트로 합치기 (추가)
        concert_text = f"""
        공연 제목: {title}
        가격: {price}
        장소: {location}
        공지사항: {notice}
        출연진: {casting}
        공연 상세정보: {detail}
        가격 정보: {price_info}
        관람 안내: {guide}
        환불 정보: {refund}
        공연 스토리: {story}
        공연 분위기: {mood}
        """
         # 텍스트 청크 분할
        chunks = self.split_text(concert_text)
        # 청크 분할 결과 확인
        for i, chunk in enumerate(chunks):
            print(f"공연 ID: {concert_id}")
            print(f"청크 번호: {i + 1}")
            print(f"청크 길이: {len(chunk)}")
            print(f"청크 내용: {chunk}")
            print("-" * 50)

        # 기존 공연 데이터에 청크 추가
        self.store[concert_id]["chunks"] = chunks

        # 실제 생성된 청크 개수 반환
        return len(chunks)


    # 공연 삭제
    def delete_concert(self, concert_id):
        if concert_id in self.store:
            self.store.pop(concert_id)
            return 1

        return 0


# 엔진 생성 함수
def create_engine(settings):
    return StubEngine(settings)
