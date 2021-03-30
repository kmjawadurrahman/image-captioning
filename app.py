import os
import pickle

import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from model import EncoderCNN, DecoderRNN


STYLE = """
<style>
img {
    max-width: 100%;
}
</style>
"""
 
def get_vocab(vocab_filepath='./vocab.pkl'):
    """Load the vocabulary from file."""
    with open(vocab_filepath, 'rb') as f:
        vocab = pickle.load(f)
        word2idx = vocab.word2idx
        idx2word = vocab.idx2word
        vocab_size = len(vocab)
    return vocab_size, word2idx, idx2word

def clean_sentence(output, idx2word):
    sentence = ' '.join([idx2word[index] for index in output])
    sentence = sentence.replace("<start> ", "")
    sentence = sentence.replace(" <end>", "")
    
    return sentence

def main():
    st.title('Image Captioning App')
    st.markdown(STYLE, unsafe_allow_html=True)
 
    file = st.file_uploader("Upload file", type=["png", "jpg", "jpeg"])
    show_file = st.empty()
 
    if not file:
        show_file.info("Please upload a file of type: " + ", ".join(["png", "jpg", "jpeg"]))
        return
 
    content = file.getvalue()

    show_file.image(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_file = 'encoder-5-batch-128-hidden-256-epochs-5.pkl'
    decoder_file = 'decoder-5-batch-128-hidden-256-epochs-5.pkl'

    embed_size = 300
    hidden_size = 256

    vocab_size, word2idx, idx2word = get_vocab()

    encoder = EncoderCNN(embed_size)
    encoder.eval()
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
    decoder.eval()

    encoder.load_state_dict(torch.load(os.path.join('./models', encoder_file)))
    decoder.load_state_dict(torch.load(os.path.join('./models', decoder_file)))

    encoder.to(device)
    decoder.to(device)

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))
    ])
    PIL_image = Image.open(file).convert('RGB')
    orig_image = np.array(PIL_image)
    image = transform_test(PIL_image)
    image = image.to(device).unsqueeze(0)
    features = encoder(image).unsqueeze(1)
    output = decoder.sample(features)

    sentence = clean_sentence(output, idx2word)
    st.info("Generated caption --> " + sentence)

    file.close()



main()