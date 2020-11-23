import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import sys


def read_article(file_name):
    file = open(file_name, "r")
    filedata = file.readlines()
    article = filedata[0].split(". ")
    sentences = []
    return filedata[0]


def generate_summary_abs(filename):
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    device = torch.device('cpu')

    text = read_article(filename)
    preprocess_text = text.strip().replace("\n","")
    t5_prepared_Text = "summarize: "+preprocess_text
    print('##################  Abstractive Text Summary  #############################')
    print('')
    print ("original text preprocessed: \n", preprocess_text)
    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
    # summmarize 
    summary_ids = model.generate(tokenized_text,
                                    num_beams=4,
                                    no_repeat_ngram_size=2,
                                    min_length=30,
                                    max_length=1000,
                                    early_stopping=True)

    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print ("\n\nSummarized Bullets: \n")
#    print ("\n\nSummarized Text: \n",output)
    print('')
    for ln in output.split('.'):
        if len(ln) > 0:
            print("* ", ln+".")
            print('')
    print('')
    print('##################  Abstractive Text Summary  #############################')
    print('')

