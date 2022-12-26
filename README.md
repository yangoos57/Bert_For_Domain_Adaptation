# Bert_For_Domain_Adaptation

- Huggingface의 Transformers, Datasets 라이브러리를 활용해 Bert Domain Adaptation을 수행
- 데이터/컴퓨터 과학 분야의 도서 데이터를 수집하여 전처리 한 뒤 Bert-Base 모델에 학습시켰음.(약 176,000개 문장)
- Domain Adpatation을 따라할 수 있도록 절차별 [상세 설명](https://yangoos57.github.io/blog/DeepLearning/paper/Electra/electra/) 및 Tutorial 제작

<br/>

### Domain Adaptation

- Pre-trained 모델을 세부 분야(ex 정치, 경제, 과학 등)에 적합한 모델로 만들기 위해 추가 학습하는 단계로서 Finetuning 이전에 수행
- Domain Adaptation 방법이 모델을 처음부터 학습하는 방법이 동일하므로 `Further pre-training`, `Continutal pre-training`으로도 불림.
- Finetuning과 Domain Adaptation에 대한 상세한 설명은 [[NLP] Further Pre-training 및 Fine-tuning 정리](https://yangoos57.github.io/blog/DeepLearning/paper/Finetuning/Finetuning/) 참고

<br/>

### 이런분들이 활용하면 좋습니다.

- Huggingface의 Trainer API를 이해하고 직접 활용하고 싶은 분
- Bert를 활용해 Domain Adapataion 또는 처음부터 모델을 학습시키고 싶으신 분
- `Trainer 커스터마이징` 또는 `callback` 함수를 활용해 학습 과정을 모니터링 하고 싶으신 분

<br/>

## Bert Domain Adaptation Tutorial With Huggingface Tutorial

- 구동환경

  ```python
    torch == 1.12.1
    pandas == 1.4.3
    transformers == 4.20.1
    datasets == 2.8.0
  ```

### 1. Bert 모델 불러오기

- [Beomi님의 kcBert](https://github.com/Beomi/KcBERT)를 베이스 모델로 활용

- 학습을 위해서 `BertForMaskedLM`로 모델을 불러와야함. `BertModel`은 encoder의 마지막단을 결과값으로 제공하기 때문

```python
from transformers import BertForMaskedLM, BertTokenizer


tokenizer = BertTokenizer.from_pretrained('beomi/kcbert-base')
model = BertForMaskedLM.from_pretrained('beomi/kcbert-base')
```

<br/>

### 2. Huggingface의 Datasets 라이브러리로 데이터 불러오기

- Huggingface의 Trainer로 모델을 학습할 예정이라면 Datasets으로 학습 자료를 불러오는 것을 추천

- pytorch의 Dataset으로 Trainer를 사용할 수 있으나 경험 상 디버깅이 상당히 번거로움.
- Trainer와 연동성이 보장된 Dataset은 간편하게 데이터를 활용할 수 있음

```python
from datasets import load_dataset

# local file을 불러오기 위해선 ('형식', '경로')를 arg로 넣어야함.
# csv 외에도 json, text, parquet, sql 형식을 지원

train = load_dataset('csv',data_files='data/book_train_128.csv')
validation = load_dataset('csv',data_files='data/book_validation_128.csv')
```

#### 불러온 데이터 확인

```python
train['train'][0]
```

    {'Unnamed: 0': 0,
     'sen': '이 책의 특징ㆍ코딩의 기초 기초수학 논리의 기초 C언어의 기초 순서도 개념ㆍ16장으로 구성하여 34시간 정도의 공부ㆍ유튜브 동영상 강의ㆍ블로그에서 모든 예제파일 다운로드 및 소프트웨어 교육 참조ㆍ공개 소프트웨어무료 컴파일러를 사용 이 책의 대상 독자ㆍ중학생 고등학생 컴퓨터관련 비전공 대학생ㆍIT 관련 취업 교육생ㆍ정보처리 기능사 산업기사 실기문제알고리즘 준비 이 책에서 다루는 내용ㆍ순서도 개념 및 코딩 작성ㆍ기초수학'}

<br/>

### 3. 데이터 토크나이징

- Trainer에 활용하기 위해선 데이터에 대한 토크나이징을 수행해야함.

- Datasets에서 제공하는 map 함수를 활용하면 간편하게 토크나이징이 가능함.

```python
def tokenize_function(examples):
    return tokenizer(examples['sen'], max_length=128, padding=True, truncation=True)

train_data_set = train['train'].map(tokenize_function)
validation_data_set = validation['train'].map(tokenize_function)
```

      0%|          | 0/175900 [00:00<?, ?ex/s]



      0%|          | 0/880 [00:00<?, ?ex/s]

```python
print(train_data_set[0])
```

    {'Unnamed: 0': 0, 'sen': '이 책의 특징ㆍ코딩의 기초 기초수학 논리의 기초 C언어의 기초 순서도 개념ㆍ16장으로 구성하여 34시간 정도의 공부ㆍ유튜브 동영상 강의ㆍ블로그에서 모든 예제파일 다운로드 및 소프트웨어 교육 참조ㆍ공개 소프트웨어무료 컴파일러를 사용 이 책의 대상 독자ㆍ중학생 고등학생 컴퓨터관련 비전공 대학생ㆍIT 관련 취업 교육생ㆍ정보처리 기능사 산업기사 실기문제알고리즘 준비 이 책에서 다루는 내용ㆍ순서도 개념 및 코딩 작성ㆍ기초수학', 'input_ids': [2, 2451, 2856, 4042, 1, 12755, 12755, 4110, 4087, 9109, 4042, 12755, 36, 4151, 4071, 4042, 12755, 21117, 4029, 1, 17474, 8455, 23335, 8432, 16502, 1, 13973, 1, 8229, 2289, 4231, 4129, 4046, 17341, 4091, 4273, 1476, 1895, 22181, 5301, 4071, 8614, 1, 1895, 22181, 5301, 4071, 4211, 4018, 3015, 4129, 4046, 4053, 4180, 9021, 2451, 2856, 4042, 12158, 1, 16264, 15513, 11315, 1664, 4203, 4239, 1, 9081, 10092, 1, 16763, 4107, 11240, 8649, 2009, 4184, 8739, 12325, 12261, 9242, 2451, 2856, 7971, 17140, 4008, 1, 10070, 1476, 3044, 4389, 1, 3], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

<br/>

### 4. Trainer 기타 기능 설정 및 학습

#### ✓ 훈련 옵션 설정(선택사항)

- 훈련에 사용되는 모든 arguments를 `TrainingArguments`를 통해 조정할 수 있음

- `logging_stetps`는 {loss,learning_rate,epoch} 정보를 몇번의 step 간격으로 수행해야할지 설정
- `evaluation_strategy`는 training 중 evaluation을 어느 때 실행해야할지 설정 'epoch'와 'step'이 있음. evaluation_strategy를 설정하지 않으면 학습 중 evaluation을 진행하지 않음.

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="test_trainer",
    per_device_eval_batch_size=64,
    per_device_train_batch_size=64,
    logging_steps=100,
    num_train_epochs=2,
    evaluation_strategy='epoch',
)

```

#### ✓ Input data 가공을 위한 Data collater 설정

- Data callter은 학습 목적에 맞게 input data를 가공하는 방법을 설정

- `DataCollatorForLanguageModeling`는 Input_data에 [MASK]를 포함하도록 가공하는 collater임. 따라서 Bert 모델 학습에 필히 설정해야함.

- Transformers는 `DataCollatorForLanguageModeling` 외에도 여러 학습 방법에 맞게 데이터를 가공하는 collater를 제공 (`DataCollatorWithPadding`, `DataCollatorForTokenClassification` 등)

```python
from transformers import DataCollatorForLanguageModeling
data_collator_BERT = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15, return_tensors="pt"
)
```

#### ✓ Callback 정의하기(선택사항)

> callback에 대한 상세한 설명은 [Huggingface로 ELECTRA 학습하기 : Domain Adaptation](https://yangoos57.github.io/blog/DeepLearning/paper/Electra/electra/) 참고

- callback은 학습 중 Trainer가 추가로 수행해야하는 Task를 정의함.

- 미리 정의된 callback을 사용하거나 아래 코드와 같이 커스텀하여 사용할 수 있음

- 아래의 `myCallback`은 100번째 Step마다 현재 epoch와 step을 출력하는 Task를 정의함.

```python
from transformers import TrainerCallback

# custom callback 만들기, 이때 TrainerCallback을 상속 받아야함.
class myCallback(TrainerCallback):
  def on_step_begin(self, args, state, control, logs=None, **kwargs):
    # step은 1회 batch 진행을 의미함. step의 시작일 때 아래의 내용을 실행
      if state.global_step % args.logging_steps == 0:
        # state는 현재 step, epoch 등 진행 상태에 대한 값을 불러옴
        # arg는 훈련 옵션으로 설정한 값을 불러옴.
          print("")
          print(
              f"{int(state.epoch)}번째 epoch 진행 중 --- {state.global_step}번째 step 결과"
          )
```

#### ✓ Custom Trainer 만들기(선택사항)

- Trainer 내부 함수를 목적에 맞게 변경할 수 있음.

- Trainer를 커스터마이징하면 아래의 예시처럼 모델 학습 경과를 시각화 할 수 있음.

```python
    0번째 epoch 진행 중 ------- 20번째 step 결과
    input 문장 : [MASK]이 출간된지 꽤 됬다고 생각하는데 실습하는데 전혀 [MASK]없습니다
    output 문장 : [책]이 출간된지 꽤 됬다고 생각하는데 실습하는데 전혀 [문제]없습니다
```

- Trainer 내부의 `compute_loss` 함수를 활용하면 input_data와 모델 학습 결과인 output_data에 접근할 수 있음

> 해당 매서드를 callback으로 구현하기에는 callback이 input_data와 output_data에 접근하기 까다롭기 때문에
>
> Trainer를 커스터마이징 하는 방법을 추천

```python
from transformers import Trainer

class customtrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    ############# 내용 추가
    def step_check(self):
        # state는 현 상태를 담는 attribute임.
        return self.state.global_step

    ### Training 단계에서
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        ############# 내용 추가
        if self.step_check() % self.args.logging_steps == 0:
            # step_check = 현 step 파악
            # args.logging_steps = argument에서 지정한 step 불러오기

            # batch 중 0 번째 위치한 문장 선택
            num = 1
            input_id = inputs.input_ids[num].reshape(-1).data.tolist()
            output_id = outputs.logits[num].argmax(dim=-1).reshape(-1).data.tolist()
            attention_mask = inputs.attention_mask[num]

            # mask가 위치한 idx 추출하기
            mask_idx = (inputs.input_ids[num] == 4).nonzero().data.reshape(-1).tolist()

            # padding 제거
            input_id_without_pad = [
                input_id[i] for i in range(len(input_id)) if attention_mask[i]
            ]
            output_id_without_pad = [
                output_id[i] for i in range(len(output_id)) if attention_mask[i]
            ]

            # id to token
            # [1:-1] [CLS,SEP] 제거
            inputs_tokens = self.tokenizer.convert_ids_to_tokens(input_id_without_pad)[
                1:-1
            ]
            outputs_tokens = self.tokenizer.convert_ids_to_tokens(
                output_id_without_pad
            )[1:-1]

            # output mask 부분 표시하기
            for i in mask_idx:
                # [CLS,SEP 위치 조정]
                outputs_tokens[i - 1] = "[" + outputs_tokens[i - 1] + "]"

            inputs_sen = self.tokenizer.convert_tokens_to_string(inputs_tokens)
            outputs_sen = self.tokenizer.convert_tokens_to_string(outputs_tokens)

            print(f"input 문장 : {''.join(inputs_sen)}")
            print(f"output 문장 : {''.join(outputs_sen)}")

        return (loss, outputs) if return_outputs else loss
```

#### ✓ Trainer 정의 및 학습 시작

- 지금까지 설정한 옵션, 데이터셋을 Trainer의 args로 활용

- 이후 train() 매서드를 통해 학습 시작

- Trainer는 매 500회 step 이후 학습된 모델을 저장하며, 학습이 중간에 중단되더라도 trainer('폴더 경로')를 통해 중단된 부분부터 새롭게 학습이 가능함.

```python
trainer = customtrainer(
    model=model,
    train_dataset=train_data_set,
    eval_dataset=validation_data_set,
    data_collator=data_collator_BERT,
    args=training_args,
    tokenizer=tokenizer,
    callbacks=[myCallback],
)

trainer.train()

# 학습 중단된 시점부터 다시 시작
# trainer.train('test_trainer/checkpoint-500)
```

<br/>

### Train 단계 전체 코드

```python
from transformers import Trainer,TrainerCallback,DataCollatorForLanguageModeling,TrainingArguments

### Training Arguments

training_args = TrainingArguments(
    output_dir="test_trainer",
    per_device_eval_batch_size=16,
    per_device_train_batch_size=16,
    logging_steps=5,
    num_train_epochs=2,
    # evaluation_strategy='epoch'
)

data_collator_BERT = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15, return_tensors="pt"
)


# custom callback 만들기, 이때 TrainerCallback을 상속 받아야함.
class myCallback(TrainerCallback):
  def on_step_begin(self, args, state, control, logs=None, **kwargs):
    # step은 1회 batch 진행을 의미함. step의 시작일 때 아래의 내용을 실행
      if state.global_step % args.logging_steps == 0:
        # state는 현재 step, epoch 등 진행 상태에 대한 값을 불러옴
        # arg는 훈련 옵션으로 설정한 값을 불러옴.
          print("")
          print(
              f"{int(state.epoch)}번째 epoch 진행 중 --- {state.global_step}번째 step 결과"
          )

class customtrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    ############# 내용 추가
    def step_check(self):
        # state는 현 상태를 담는 attribute임.
        return self.state.global_step

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        ############# 내용 추가
        if self.step_check() % self.args.logging_steps == 0:
            # step_check = 현 step 파악
            # args.logging_steps = argument에서 지정한 step 불러오기

            # batch 중 0 번째 위치한 문장 선택
            num = 1
            input_id = inputs.input_ids[num].reshape(-1).data.tolist()
            output_id = outputs.logits[num].argmax(dim=-1).reshape(-1).data.tolist()
            attention_mask = inputs.attention_mask[num]

            # mask가 위치한 idx 추출하기
            mask_idx = (inputs.input_ids[num] == 4).nonzero().data.reshape(-1).tolist()

            # padding 제거
            input_id_without_pad = [
                input_id[i] for i in range(len(input_id)) if attention_mask[i]
            ]
            output_id_without_pad = [
                output_id[i] for i in range(len(output_id)) if attention_mask[i]
            ]

            # id to token
            # [1:-1] [CLS,SEP] 제거
            inputs_tokens = self.tokenizer.convert_ids_to_tokens(input_id_without_pad)[
                1:-1
            ]
            outputs_tokens = self.tokenizer.convert_ids_to_tokens(
                output_id_without_pad
            )[1:-1]

            # output mask 부분 표시하기
            for i in mask_idx:
                # [CLS,SEP 위치 조정]
                outputs_tokens[i - 1] = "[" + outputs_tokens[i - 1] + "]"

            inputs_sen = self.tokenizer.convert_tokens_to_string(inputs_tokens)
            outputs_sen = self.tokenizer.convert_tokens_to_string(outputs_tokens)

            print(f"input 문장 : {''.join(inputs_sen)}")
            print(f"output 문장 : {''.join(outputs_sen)}")

        return (loss, outputs) if return_outputs else loss


trainer = customtrainer(
    model=model.to(device),
    train_dataset=train_data_set,
    eval_dataset=validation_data_set,
    data_collator=data_collator_BERT,
    args=training_args,
    tokenizer=tokenizer,
    callbacks=[myCallback],
)

trainer.train()

```

    PyTorch: setting up devices
    The default value for the training argument `--report_to` will change in v5 (from all installed integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as now. You should start updating your code and make this info disappear :-).
    The following columns in the training set don't have a corresponding argument in `BertForMaskedLM.forward` and have been ignored: sen, Unnamed: 0. If sen, Unnamed: 0 are not expected by `BertForMaskedLM.forward`,  you can safely ignore this message.
    /Users/yangwoolee/.pyenv/versions/3.9.1/lib/python3.9/site-packages/transformers/optimization.py:306: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
      warnings.warn(
    ***** Running training *****
      Num examples = 10
      Num Epochs = 2
      Instantaneous batch size per device = 16
      Total train batch size (w. parallel, distributed & accumulation) = 16
      Gradient Accumulation steps = 1
      Total optimization steps = 2



      0%|          | 0/2 [00:00<?, ?it/s]



    0번째 epoch 진행 중 --- 0번째 step 결과
    input 문장 : 머신러 [MASK] [MASK]문서를 떼고 실무 [MASK] 활용하려는 개발자 대학 [MASK] 배운 머신러닝 [MASK] 제품에 적용하려는 주니어 개발 [MASK] 소프트웨어 개발자는 아니지만 머신러닝 시스템 [MASK] 기술적인 [MASK]에 흥미가 있는 비즈니스 담당자머신러닝 알고리즘은 이미 다른 책에서 [MASK] 다루고 있으니 이 책에서는 머신러닝 프로 [MASK]트를 처음 시작하는 방법 기존 시스템에 머신러 [MASK]을 통합하는 방법 [MASK] [MASK]러닝에 사용할 데이터 [MASK] 수 [MASK]하는 방법 등 실무에 [MASK]용한 내용을 중점 물론이고 다룬다
    output 문장 : 머신러 [##닝] [시험]문서를 떼고 실무 [##를] 활용하려는 개발자 대학 [##에서]을 머신러닝 [##을] 제품에 적용하려는 주니어 개발 [##자] 소프트웨어 개발자는 아니지만 머신러닝 시스템 [등] 기술적인 [지식]에 흥미가 있는 비즈니스 담당자머신러닝 알고리즘은 이미 다른 책에서 [직접] 다루고 있으니 이 책에서는 머신러닝 프로 [##젝]트를 처음 시작하는 방법 기존 시스템에 머신러 [##닝]을 통합하는 방법 [또는] [##신]러닝에 사용할 데이터 [##를] 수 [##집]하는 방법 등 실무에 [유]용한 내용을 중점적으로 다룬다




    Training completed. Do not forget to share your model on huggingface.co/models =)




    {'train_runtime': 5.0572, 'train_samples_per_second': 3.955, 'train_steps_per_second': 0.395, 'train_loss': 3.2832515239715576, 'epoch': 2.0}





    TrainOutput(global_step=2, training_loss=3.2832515239715576, metrics={'train_runtime': 5.0572, 'train_samples_per_second': 3.955, 'train_steps_per_second': 0.395, 'train_loss': 3.2832515239715576, 'epoch': 2.0})
