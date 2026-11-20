
# 임시 RAG 엔진
# MongoDB Atlas 연결 전까지 공연 데이터를 메모리에 저장

class StubEngine:
    name = "stub"
    min_score = 0.0

    def __init__(self, settings):
        self.store = {}

    # 공연 색인
    def index_concert(self, concert_id, title, price, location):
        self.store[concert_id] = {
            "type": "CONCERT",
            "concert_id": concert_id,
            "title": title,
            "price": price,
            "location": location
        }

        return 1

    # 공연 삭제
    def delete_concert(self, concert_id):
        if concert_id in self.store:
            self.store.pop(concert_id)
            return 1

        return 0


# 엔진 생성 함수
def create_engine(settings):
    return StubEngine(settings)
