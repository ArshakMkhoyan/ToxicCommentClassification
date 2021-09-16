from flask import Flask, request, jsonify
from transformers import BertTokenizer
from utils import BertClassifier, clean_text
import torch

app = Flask(__name__)

model = BertClassifier()
model.load_state_dict(torch.load('bert_1_epoch.pth'))
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


@app.route("/get_sentence_toxicity", methods=['POST'])
def get_sentence_toxicity():
    request_data = request.get_json()
    sentence = request_data['sentence']
    sentence = clean_text(sentence)
    encoded_sentence = tokenizer.encode_plus(
        text=sentence,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True
    )
    input_id = torch.unsqueeze(torch.tensor(encoded_sentence['input_ids']), 0)
    attention_mask = torch.unsqueeze(torch.tensor(encoded_sentence['attention_mask']), 0)
    logits = model(input_id, attention_mask)[0]
    probs = logits.sigmoid().cpu().detach().numpy()
    probs = [round(float(prob), 3) for prob in probs]

    return jsonify(toxic=probs[0], severe_toxic=probs[1], obscene=probs[2], threat=probs[3], insult=probs[4],
                   identity_hate=probs[5])


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=12000)
