# 📖 KoBERT 기계독해 학습 (kobert_question_answering)

> **본문과 질문을 주면 본문 안에서 정답 구간을 찾아냅니다.**
> **Project Date:** 2022.11
> **Platform:** Google Colab (TPU) · TensorFlow · Hugging Face Transformers

캡스톤 프로젝트를 진행하던 중 한국어 BERT를 공부하며 정리한 **기계독해(Question Answering) 학습 노트북**입니다. 공개된 예제를 따라 **KorQuAD 1.0** 으로 한국어 BERT를 미세 조정하고, 본문과 질문을 넣었을 때 모델이 예측한 정답을 실제 정답과 나란히 출력해 확인했습니다.

직접 기획한 프로젝트가 아니라 **이미 있는 프로젝트를 공부하고 이해한 기록**이며, 노트북에서 불러오는 사전학습 모델은 `klue/bert-base`입니다.

---

## 📖 프로젝트 개요

기계독해는 본문과 질문을 함께 입력받아, 본문 안에서 **정답이 시작하는 토큰과 끝나는 토큰의 위치**를 예측하는 문제입니다.
이 저장소는 그 과정을 데이터 준비부터 예측 출력까지 노트북 한 개로 따라가며 익힌 결과물입니다.

* **데이터 준비부터 예측까지 한 흐름** — KorQuAD 내려받기 → 정답 위치 정리 → 토큰화 → 학습 → 예측 출력
* **정답 위치 두 개를 맞히는 구조** — BERT 출력 위에 Dense 층 하나를 얹어 시작 위치와 끝 위치의 확률을 각각 계산
* **다른 분야로 넓힐 여지** — 같은 형식의 JSON을 만들어 넣으면 역사·사회 등 다른 분야의 문제도 다룰 수 있을 것으로 보았습니다

### 💡 주요 기능

| 기능 | 설명 |
|---|---|
| 📥 **데이터 분리** | KorQuAD JSON을 본문·질문·답변 목록으로 나누고, 답변의 끝 위치(`answer_end`)를 계산해 추가 |
| ✂️ **토큰화와 위치 변환** | `klue/bert-base` 토크나이저로 본문과 질문을 묶어 토큰화하고, 글자 단위 정답 위치를 토큰 위치로 변환 |
| 🧹 **샘플 정리** | 정답의 토큰 위치를 찾지 못한 샘플은 학습·평가에서 제외 |
| 🧠 **미세 조정** | `TFBertModel` + Dense(2) 모델을 TPU 분산 전략으로 3 에폭 학습 |
| 🔍 **예측 확인** | 검증 데이터의 본문·질문·정답·예측을 한 묶음씩 출력해 눈으로 대조 |

---

## 🏗️ 아키텍처 — 학습 파이프라인

**노트북 실행 흐름 (KorQuAD → 정답 예측)**

1. `KorQuAD_v1.0_train.json` · `KorQuAD_v1.0_dev.json`을 내려받아 본문·질문·답변으로 분리
2. 답변 글자 수로 끝 위치를 계산하고, 본문에서 잘라낸 글자가 정답과 같은지 `assert`로 검증
3. 본문과 질문을 한 쌍으로 토큰화한 뒤 정답의 시작·끝 글자 위치를 토큰 위치로 변환
4. 토큰 위치를 찾지 못한 샘플을 제거하고 입력(`input_ids`, `attention_mask`)과 정답(시작·끝 위치)으로 구성
5. `TPUStrategy` 범위 안에서 모델을 만들고 Adam(학습률 5e-5) · 배치 16 · 3 에폭 학습
6. 검증 샘플을 모델에 넣어 시작·끝 확률이 가장 큰 위치를 골라 예측 답을 복원

> 시작 위치와 끝 위치를 각각 토큰 분류 문제로 보고 `SparseCategoricalCrossentropy` 두 개의 합을 손실로 사용했습니다.

---

## 📊 데이터 규모와 실측

| 단계 | 결과 |
|---|---|
| **학습 데이터** | KorQuAD 1.0 train — 본문·질문·답변 **60,407쌍** |
| **검증 데이터** | KorQuAD 1.0 dev — 본문·질문·답변 **5,774쌍** |
| **학습 손실** | 에폭 1 **1.1412** → 에폭 2 **0.6208** → 에폭 3 **0.4418** |
| **학습 시간** | Colab TPU에서 에폭당 3,759 스텝, 약 440 ~ 536초 |
| **예측 확인** | 검증 샘플 100건을 출력했을 때 정답과 예측 문자열이 같은 경우 **82건** (노트북 출력 기준, 정식 EM·F1 평가는 아님) |

---

## 🛠️ 기술 스택

| 영역 | 스택 |
|---|---|
| **언어·환경** | <img src="https://img.shields.io/badge/Python_3.7-3776AB?style=flat-square&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/Google_Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=black"/> |
| **모델** | <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white"/> <img src="https://img.shields.io/badge/Transformers_4.24-FFD21E?style=flat-square&logo=huggingface&logoColor=black"/> `klue/bert-base` (`BertTokenizerFast` · `TFBertModel`) |
| **데이터** | KorQuAD 1.0 (train · dev) |

---

## 💥 트러블 슈팅

### 1. 질문이 `[PAD]`로 출력되는 경우
* **현상:** 예측 결과를 출력할 때 질문 자리에 질문 대신 `[PAD]`가 이어서 나오는 샘플이 있음. 출력한 100건 중 1건이 해당.
* **원인:** 본문과 질문을 `'[SEP] '`(뒤에 공백 포함) 기준으로 잘라 출력하는데, 해당 샘플은 질문이 작은따옴표로 시작해 복원된 문장에서 `[SEP]'`처럼 공백 없이 붙음. 그래서 질문이 본문 쪽에 붙어 나오고, 질문 자리에는 뒤쪽 패딩이 출력됨. 정답과 예측은 정상적으로 출력됨.
* **해결:** 수정하지 못한 채로 남아 있음.

---

## 📂 디렉터리 구조

```text
kobert_question_answering/
├── kobert_question_answering.ipynb  # 데이터 준비 · 학습 · 예측 출력 전 과정 (Colab 노트북)
└── README.md
```

---

## 🚀 실행 방법

**요구 사항:** Google Colab · 런타임 유형 **TPU**

1. `kobert_question_answering.ipynb`를 Google Colab에서 엽니다.
2. 런타임 유형을 TPU로 바꿉니다. (노트북이 `COLAB_TPU_ADDR` 환경 변수로 TPU에 연결하므로 Colab 밖에서는 그대로 실행되지 않습니다.)
3. 위에서부터 차례로 실행합니다. `transformers` 설치와 KorQuAD 데이터 내려받기는 노트북 안의 셀이 처리합니다.

---

## 📝 회고

**프로젝트 후기**

- 직접 생각한 프로젝트가 아니라 이미 있는 프로젝트를 공부하고 이해한 것이어서 성취감이 다른 프로젝트보다 부족했습니다.
- 그러나 흥미로운 주제를 알 수 있어서 프로젝트를 공부하는 것이 재미있었습니다.
- 이 프로젝트를 발전시켜 시험 문제를 직접 출제해 보고 싶다는 생각이 들었습니다.

**아쉬웠던 점**

- 질문이 `[PAD]`로 나온 경우가 있는데, 이 경우를 수정하지 못했습니다.
- Google Colab 환경이 아닌 Jupyter 환경에서도 실험해 보고 싶었는데, 오류를 고치는 데에만 많은 시간이 걸려 진척도가 높지 않았습니다.

## 참고 자료

- [위키독스 — 참고한 예제 문서](https://wikidocs.net/156772)
- [SKTBrain/KoBERT](https://github.com/SKTBrain/KoBERT#release)
- [KorQuAD](https://korquad.github.io/) — 학습·검증 데이터
