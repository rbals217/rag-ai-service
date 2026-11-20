from fastapi import FastAPI

# 프로그램의 메인 역할을 함
# 1. 서버 초기화: 설정, RAG엔진 최초 생성
# 2. 서비스 인증: Spring Boot가 전달하는 서비스 토큰 검증
# 3. 공연 색인: /index/concert로 전달받은 공연 저장
# 4. 공연 삭제: /index/concert/{concert_id}로 색인 데이터 삭제
# 5. AI 질의: /query로 전달받은 질문 처리

import logging
import secrets
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from engine import create_engine
from schemas import (
    ConcertIndexRequest,
    ConcertIndexResponse,
    ConcertDeleteResponse,
    QueryResponse,
    QueryRequest
)
from settings import Settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
log = logging.getLogger("rag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 한 번: 설정 → 엔진
    settings = Settings()
    engine = create_engine(settings)
    app.state.settings = settings
    app.state.engine = engine

    log.info("RAG 엔진 준비 완료: %s", engine.name)
    yield

    client = getattr(engine, "client", None)
    if client is not None:
        client.close()


app = FastAPI(title="TicketPass Concert RAG AI Service", lifespan=lifespan)


# 외부에서 전달된 서비스 토큰이 서버에 설정된 토큰과 같은지 확인
def verify_token(request: Request, x_service_token: str = Header(default="")) -> None:
    expected = request.app.state.settings.service_token
    if not secrets.compare_digest(x_service_token, expected):
        raise HTTPException(status_code=401, detail="서비스 토큰이 올바르지 않습니다.")


@app.get("/health")
def health():
    return {"status": "OK"}


# 공연 등록 및 수정 시 Spring Boot가 호출하는 API
@app.post("/index/concert", response_model=ConcertIndexResponse, dependencies=[Depends(verify_token)])
async def index_concert(req: ConcertIndexRequest, request: Request):
    engine = request.app.state.engine
    try:
        n = await run_in_threadpool(
            engine.index_concert,
            req.concert_id,
            req.title,
            req.price,
            req.location,
            req.notice,
            req.casting,
            req.detail,
            req.price_info,
            req.guide,
            req.refund,
            req.story,
            req.mood
        )
    except Exception as e:
        log.exception("공연 색인 실패")
        raise HTTPException(status_code=502, detail=f"공연 색인 실패: {type(e).__name__}") from e

    return ConcertIndexResponse(concert_id=req.concert_id, chunks=n)


# 공연 삭제 시 호출
@app.delete("/index/concert/{concert_id}", response_model=ConcertDeleteResponse, dependencies=[Depends(verify_token)])
async def delete_concert(concert_id: int, request: Request):
    engine = request.app.state.engine
    try:
        n = await run_in_threadpool(engine.delete_concert, concert_id)
    except Exception as e:
        log.exception("공연 색인 삭제 실패")
        raise HTTPException(status_code=502, detail=f"공연 삭제 실패: {type(e).__name__}") from e

    return ConcertDeleteResponse(concert_id=concert_id, deleted=n)


# AI 질문 처리 API
@app.post("/query", response_model=QueryResponse, dependencies=[Depends(verify_token)])
async def query(req: QueryRequest, request: Request):
    return QueryResponse(
        answer="아직 AI 검색 기능이 구현되지 않았습니다.",
        abstained=True,
        sources=[],
        trace=["query_not_implemented"]
    )