''' Convenience functions for handling BERT '''
import torch
#import pytorch_pretrained_bert as bert
from transformers import AutoTokenizer, AutoModel
#from transformers import BertModel
#from tokenization_kobert import KoBertTokenizer


def load_model(version='beomi/KcELECTRA-base'):
    ''' Load BERT model and corresponding tokenizer '''
    #tokenizer = bert.KoBertTokenizer.from_pretrained(version)
    #model = bert.BertModel.from_pretrained(version)
    tokenizer = AutoTokenizer.from_pretrained(version)
    model = AutoModel.from_pretrained(version)
    #model_config = BertConfig.from_pretrained(version, output_hidden_states=True)
    #model = BertModel.from_pretrained(version, config=model_config)
    model.eval()

    return model, tokenizer


def encode(model, tokenizer, texts):
    ''' Use tokenizer and model to encode texts '''
    encs = {}
    for text in texts:
        #print(text)
        #print(type(text))
        # https://huggingface.co/transformers/main_classes/output.html#basemodeloutputwithpooling
        # https://huggingface.co/transformers/main_classes/output.html#basemodeloutputwithpastandcrossattentions 
        tokenized = tokenizer(text, return_tensors="pt")
        outputs = model(**tokenized)
        last_hidden_state = outputs.last_hidden_state
        #enc = last_hidden_state[0]
        enc = last_hidden_state[:, 0, :]
        #print('enc:', enc)
        #print('enc.detach().view(-1).numpy():', enc.detach().view(-1).numpy())

        #tokenized = tokenizer.tokenize(text)
        #indexed = tokenizer.convert_tokens_to_ids(tokenized)
        #segment_idxs = [0] * len(tokenized)
        #tokens_tensor = torch.tensor([indexed])
        #segments_tensor = torch.tensor([segment_idxs])
        #enc = model(tokens_tensor, segments_tensor)
        #enc, _ = model(tokens_tensor, segments_tensor)
        #enc = enc[:, 0, :]  # extract the last rep of the first input
        #print(model(tokens_tensor, segments_tensor))
        # ref: https://pytorch.org/hub/huggingface_pytorch-transformers/ 
        #print('outputs:', outputs)
        #enc = outputs[0]
        # reference: https://code.ihub.org.cn/projects/763/repository/commit_diff?changeset=f31154cb9df44b9535bd21eb5962e7a91711e9d1 
        encs[text] = enc.detach().view(-1).numpy()
        #print(encs[text])
    return encs