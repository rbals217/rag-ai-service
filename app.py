from contextlib import asynccontextmanager
import logging
import secrets
# 프로그램의 메인 역할을 함
# 1. 서버 초기화: 설정, RAG엔진, LangGraph를 최초 생성
# 2. 서비스 인증: Spring Boot가 전달하는 서비스 트큰 검증
# 3. 게시글 색인: /index로 전달받은 게시글 저장
# 4. 게시글 삭제: /index/{board_id}로 색인 데이터 삭제
# 5. AI 질의: /query로 전달받은 질문 처리
from fastapi import FastAPI, Depends, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from engine import create_engine
from graph import build_graph, run_graph
from schemas import (
    ConcertIndexRequest,
    ConcertIndexResponse,
    ConcertDeleteResponse,
    QueryResponse,
    QueryRequest
)
from settings import Settings


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)

log = logging.getLogger("rag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    engine = create_engine(settings)

    # LangGraph 생성
    graph = build_graph(engine, settings)

    app.state.settings = settings
    app.state.engine = engine
    app.state.graph = graph

    log.info("RAG 엔진 및 그래프 준비 완료: %s", engine.name)

    try:
        yield
    finally:
        # 엔진에 close()가 구현되어 있을 때만 실행
        close = getattr(engine, "close", None)

        if callable(close):
            close()


app = FastAPI(
    title="TicketPass Concert RAG AI Service",
    lifespan=lifespan
)


# 서비스 토큰 검증
def verify_token(
    request: Request,
    x_service_token: str = Header(default="")
) -> None:

    expected = request.app.state.settings.service_token

    if not secrets.compare_digest(x_service_token, expected):
        raise HTTPException(
            status_code=401,
            detail="서비스 토큰이 올바르지 않습니다."
        )


# 서버 상태 확인
@app.get("/health")
def health(request : Request):
    return {"status": "OK","engine":request.app.state.engine.name}
@app.post(
    "/index/vector",
    dependencies=[Depends(verify_token)]
)
async def create_vector_index(request: Request):

    engine = request.app.state.engine

    if not hasattr(engine, "create_vector_index"):
        raise HTTPException(
            status_code=400,
            detail="Atlas 엔진에서만 사용할 수 있습니다."
        )

    try:
        result = await run_in_threadpool(
            engine.create_vector_index
        )

        return result

    except Exception as e:
        log.exception("Vector Search 인덱스 생성 실패")

        raise HTTPException(
            status_code=500,
            detail=f"인덱스 생성 실패: {type(e).__name__}"
        ) from e
# Vector Search 인덱스 상태 확인
@app.get(
    "/index/vector/status",
    dependencies=[Depends(verify_token)]
)
async def vector_index_status(request: Request):

    engine = request.app.state.engine

    if engine.name != "atlas":
        raise HTTPException(
            status_code=400,
            detail="Vector Search 상태 확인은 Atlas 엔진에서만 가능합니다."
        )

    try:
        result = await run_in_threadpool(
            engine.get_vector_index_status
        )

        return result

    except Exception as e:
        log.exception("Vector Search 인덱스 상태 확인 실패")

        raise HTTPException(
            status_code=502,
            detail=f"Vector Search 상태 확인 실패: {type(e).__name__}"
        ) from e
# 공연 색인
@app.post(
    "/index/concert",
    response_model=ConcertIndexResponse,
    dependencies=[Depends(verify_token)]
)
async def index_concert(req: ConcertIndexRequest, request: Request):

    engine = request.app.state.engine

    try:
        log.info(
            "[RAG 수신 데이터] ID=%s, 제목=%s",
            req.concert_id,
            req.title
        )
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

        raise HTTPException(
            status_code=502,
            detail=f"공연 색인 실패: {type(e).__name__}"
        ) from e

    return ConcertIndexResponse(
        concert_id=req.concert_id,
        chunks=n
    )


# 공연 색인 삭제
@app.delete(
    "/index/concert/{concert_id}",
    response_model=ConcertDeleteResponse,
    dependencies=[Depends(verify_token)]
)
async def delete_concert(concert_id: int, request: Request):

    engine = request.app.state.engine

    try:
        n = await run_in_threadpool(
            engine.delete_concert,
            concert_id
        )

    except Exception as e:
        log.exception("공연 색인 삭제 실패")

        raise HTTPException(
            status_code=502,
            detail=f"공연 삭제 실패: {type(e).__name__}"
        ) from e

    return ConcertDeleteResponse(
        concert_id=concert_id,
        deleted=n
    )


# AI 질문 및 검색
@app.post(
    "/query",
    response_model=QueryResponse,
    dependencies=[Depends(verify_token)]
)
async def query(req: QueryRequest, request: Request):

    graph = request.app.state.graph

    try:
        # LangGraph 실행
        result = await run_in_threadpool(
            run_graph,
            graph,
            req.question,
            req.concert_ids,
            req.top_k
        )

    except Exception as e:
        log.exception("공연 질의 처리 실패")

        raise HTTPException(
            status_code=502,
            detail=f"공연 질의 처리 실패: {type(e).__name__}"
        ) from e

    return QueryResponse(
        answer=result["answer"],
        abstained=result["abstained"],
        sources=result.get("sources", []),
        trace=result.get("trace", [])
    )