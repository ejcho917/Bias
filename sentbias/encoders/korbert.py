# maybe for korbert copy and paste the tokenization.py file in /src 

# from here was the original file

def load_model(version='003_bert_eojeol_pytorch'):
    ''' Load KorBERT model and corresponding tokenizer '''
    #tokenizer = bert.KoBertTokenizer.from_pretrained(version)
    #model = bert.BertModel.from_pretrained(version)
    tokenizer = BertTokenizer.from_pretrained(version)
    model = BertModel.from_pretrained(version)
    #model_config = BertConfig.from_pretrained(version, output_hidden_states=True)
    #model = BertModel.from_pretrained(version, config=model_config)
    model.eval()

    return model, tokenizer


def encode(model, tokenizer, texts):
    ''' Use tokenizer and model to encode texts '''
    encs = {}
    for text in texts:
        tokenized = tokenizer.tokenize(text)
        indexed = tokenizer.convert_tokens_to_ids(tokenized)
        segment_idxs = [0] * len(tokenized)
        tokens_tensor = torch.tensor([indexed])
        segments_tensor = torch.tensor([segment_idxs])
        enc, _ = model(tokens_tensor, segments_tensor)
        enc = enc[:, 0, :]  # extract the last rep of the first input
        #print('enc:', enc)
        #print('enc.detach().view(-1).numpy():', enc.detach().view(-1).numpy())
        encs[text] = enc.detach().view(-1).numpy()
        
        #tokenized = tokenizer(text, return_tensors="pt")
        #outputs = model(**tokenized)
        #last_hidden_state = outputs.last_hidden_state
        #enc = last_hidden_state[0]
        #encs[text] = enc.detach().view(-1).numpy()
    return encs