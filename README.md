# Bert로 Domain Adaptation 수행하기

### 프로젝트 소개
- Huggingface의 Transformers, Datasets 라이브러리를 활용해 Bert Domain Adaptation을 수행
- 데이터/컴퓨터 과학 분야의 도서 데이터를 수집하여 전처리 한 뒤 Bert-Base 모델에 학습(약 176,000개 문장)
- 🤗Transoformer를 활용해 Domain Adpatation을 수행하는 방법에 대한 Tutorial 제작

<br/>

### Domain Adaptation이란?

- Pre-trained 모델을 세부 분야(ex 정치, 경제, 과학 등)에 적합한 모델로 만들기 위해 추가 학습하는 단계로서 Fine-tuning 이전에 수행
- Domain Adaptation 수행 방법은 모델을 처음부터 학습하는 방법과 동일하기 때문에 `Further pre-training`, `Continutal pre-training`이라는 용어로도 불림.
- Fine-tuning과 Domain Adaptation에 대한 소개는 [[NLP] Further Pre-training 및 Fine-tuning 정리](https://yangoos57.github.io/blog/DeepLearning/paper/Finetuning/Finetuning/) 참고

<br/>

### 이런분들이 활용하면 좋습니다.

- Huggingface의 Trainer API를 이해하고 직접 활용하고 싶은 분
- Bert를 활용해 Domain Adapataion 또는 처음부터 모델을 학습시키고 싶으신 분
- `Trainer 커스터마이징` 또는 `callback` 함수를 활용해 학습 과정을 모니터링 하고 싶으신 분

<br/>

