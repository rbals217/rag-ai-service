
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


# Spring Boot와 FastAPI 간 데이터 형식
class CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True
    )


# 공연 색인 요청
class ConcertIndexRequest(CamelModel):
    concert_id: int = Field(gt=0)
    title: str = Field(min_length=1, max_length=200)
    price: int | None = Field(default=None, ge=0)
    location: str | None = Field(default=None, max_length=200)

    notice : str | None= None
    casting : str | None= None
    detail : str | None= None
    price_info : str | None = None
    guide : str | None = None
    refund : str | None = None
    story : str | None = None
    mood : str | None = None        # None은 모든 값 다 받겠다는 소리

# 공연 색인 결과
class ConcertIndexResponse(CamelModel):
    concert_id: int
    chunks: int


# 공연 삭제 결과
class ConcertDeleteResponse(CamelModel):
    concert_id: int
    deleted: int


# AI 질문 요청
class QueryRequest(CamelModel):
    question: str = Field(min_length=1, max_length=500)
    concert_ids: list[int] = Field(
        default_factory=list,
        max_length=10_000
    )
    top_k: int = Field(default=4, ge=1, le=10)

# 검색 출처
class Source(CamelModel):
    concert_id: int
    title: str
    field_name: str
    snippet: str
    score: float


# AI 답변
class QueryResponse(CamelModel):
    answer: str
    abstained: bool
    sources: list[Source] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)
