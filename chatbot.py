# pip install transformers==4.38.2

from transformers import AutoTokenizer,AutoModelForSeq2SeqLM

model_name = "facebook/blenderbot-400M-distill"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


convertsation_history = []

while True:
    history_string = "\n".join(convertsation_history)

    input_text = input("> ")

    inputs = tokenizer.encode_plus(history_string,input_text,return_tensors="pt")
    #print(inputs)

    tokenizer.pretrained_vocab_files_map

    outputs = model.generate(**inputs)
    #print(outputs)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print(response)

    convertsation_history.append(input_text)
    convertsation_history.append(response)
    print(convertsation_history)

