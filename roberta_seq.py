from lyc.utils import get_model, get_tokenizer
from lyc.data import get_hf_ds_scripts_path, get_tokenized_ds, get_dataloader
import sys
from lyc.train import get_base_hf_args, HfTrainer
from lyc.eval import tagging_eval_for_trainer, write_predict_to_file, eval_with_weights
from transformers import RobertaForTokenClassification, DataCollatorForTokenClassification, Trainer
from transformers.integrations import TensorBoardCallback
import os
import datasets
import torch

if __name__ == '__main__':

    model_name, data_dir, = sys.argv[1:]
    save_folder = '/vol/research/nlg/frame_finder/'
    output_dir = os.path.join(save_folder, 'checkpoints/roberta_seq/')
    logging_dir = os.path.join(save_folder, 'logs/')
    prediction_output_file = os.path.join(output_dir, 'prediction_output.csv')

    tokenizer = get_tokenizer(model_name, add_prefix_space=True)

    p = get_hf_ds_scripts_path('vua20')
    data_files={'train': os.path.join(data_dir, 'train.tsv'), 'test': os.path.join(data_dir, 'test.tsv')}
    ds = get_tokenized_ds(p, tokenizer, tokenize_func='tagging', \
        tokenize_cols=['tokens'], tagging_cols={'is_target':0, 'labels':-100}, \
        data_files=data_files, name='combined', batched=False)
    ds = ds.rename_column('is_target', 'token_type_ids')

    # dl = get_dataloader(ds, cols=['attention_mask', 'input_ids', 'labels'])
    args = get_base_hf_args(
        output_dir=output_dir,
        logging_steps=1,
        logging_dir = logging_dir,
        lr=5e-5,
        train_batch_size=24,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer, max_length=128)
    model = get_model(RobertaForTokenClassification, model_name, num_labels = 2, type_vocab_size=2)
    # model.roberta.embeddings.token_type_embeddings = torch.nn.Embedding(2, 768)
    # model._init_weights(model.roberta.embeddings.token_type_embeddings)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[TensorBoardCallback()],
        compute_metrics=tagging_eval_for_trainer
    )

    # trainer.train()
    # trainer.save_model()

    # result = trainer.evaluate()
    # print(result)

    # test_ds = datasets.Dataset.from_dict(ds['test'][:10])
    pred_out = trainer.predict(ds['test'])
    write_predict_to_file(pred_out, out_file=prediction_output_file)
    result = eval_with_weights(pred_out, ds['test']['token_type_ids'])
    print(result)
