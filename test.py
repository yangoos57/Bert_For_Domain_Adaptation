from transformers import TrainingArguments, TrainerCallback

training_args = TrainingArguments(
    output_dir="test_trainer",
    per_device_eval_batch_size=16,
    per_device_train_batch_size=16,
    logging_steps=5,
    num_train_epochs=2,
)

data_collator_BERT = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15, return_tensors="pt"
)


class myCallback(TrainerCallback):
    """
    A bare :class:`~transformers.TrainerCallback` that just prints the logs.
    """

    def on_step_begin(self, args, state, control, logs=None, **kwargs):

        # 함수 이름.. 언제 시작할지
        # log는 설정할 때마다
        # arg,state,control은 참고할 수 있는 attribute의 경우임.
        # 근데 내가 필요한건 input
        if state.global_step % args.logging_steps == 0:
            print("")
            print(
                f"{int(state.epoch)}번째 epoch 진행 중 ------- {state.global_step}번째 step 결과"
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
            # args.logging_steps = argument에서 지정한 step 불러오가

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
    model=model,
    train_dataset=train_data_set,
    eval_dataset=validation_data_set,
    data_collator=data_collator_BERT,
    args=training_args,
    tokenizer=tokenizer,
    callbacks=[myCallback],
)

trainer.train()
