from lyc.data import get_hf_ds_scripts_path, get_tokenized_ds, get_dataloader
from lyc.utils import get_model, get_tokenizer
from transformers import RobertaForTokenClassification
import sys
from tqdm import tqdm
import torch
import json

if __name__ == '__main__':

    model_name, = sys.argv[1:]

    tokenizer = get_tokenizer(model_name)
    
    script_path = get_hf_ds_scripts_path('vua20')
    ds = get_tokenized_ds(script_path, tokenizer, max_length=128, tokenize_func='general', tokenize_cols=['tokens'],
        is_split_into_words=True, return_word_ids=True,
        data_files={
            'train':'/Users/liyucheng/projects/acl2021-metaphor-generation-conceptual-main/EM/data/VUA20/train.tsv',
            'test': '/Users/liyucheng/projects/acl2021-metaphor-generation-conceptual-main/EM/data/VUA20/test.tsv'})
    
    dl = get_dataloader(ds['test'], batch_size=8, cols=['input_ids', 'attention_mask', 'word_index', 'words_ids', 'label'])
    model = get_model(RobertaForTokenClassification, model_name)

    with open('frame_labels.json', encoding='utf-8') as f:
        label2id = json.load(f)
    
    id2label = {v:k for k,v in label2id.items()}
    file = open('vua_frame.csv', 'w', encoding='utf-8')

    for index, batch in enumerate(tqdm(dl)):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        word_index = batch.pop('word_index')
        word_ids = batch.pop('words_ids')
        mlabels = batch.pop('label')
        outputs = model(**batch)
        logits = outputs.logits
        labels = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)

        for words, label, mlabel, word_idx, word_id in zip(batch['input_ids'], labels, mlabels, word_index, word_ids):
            words = tokenizer.convert_ids_to_tokens(words)
            label = [id2label[i.item()] for i in label]
            words = [word.strip('Ġ') for word in words]
            word_id = [str(i.item()) if i is not None else str(i) for i in word_id]
            file.write('token\t' + '\t'.join(words)+'\n')
            file.write('frame\t' + '\t'.join(label)+'\n')
            file.write('word_id\t' + '\t'.join(word_id)+'\n')
            file.write('target index\t' + str(word_idx.item())+'\n')
            file.write('metaphor label\t' + str(mlabel.item())+'\n')

        if index>200:
            break
    
    file.close()
        # print(outputs)
        # break