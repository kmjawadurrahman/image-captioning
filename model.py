import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)
        self.lstm = nn.LSTM(
                        input_size=self.embed_size,
                        hidden_size=self.hidden_size,
                        num_layers=self.num_layers,
                        batch_first=True
                    )
        self.dropout = nn.Dropout(0.3)
        self.fc_output = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)
    
    def forward(self, features, captions):
        # leave out last token since model does not need to learn how to predict word following <end> token
        captions = captions[:, :-1]
      
        embedded_captions = self.embed(captions)
        
        # concatenate image features and embedded captions
        inputs = torch.cat((features.unsqueeze(1), embedded_captions), dim=1)
        
        outputs, _ = self.lstm(inputs)
        outputs = self.dropout(outputs)
        outputs = self.fc_output(outputs)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        output = []
        batch_size = inputs.shape[0]
        hidden = (torch.randn(1, batch_size, 256).to(inputs.device), 
                  torch.randn(1, batch_size, 256).to(inputs.device))
        while True:
            lstm_out, hidden = self.lstm(inputs, hidden)
            outputs = self.fc_output(lstm_out)
            outputs = outputs.squeeze(1)
            _, max_pred_index = torch.max(outputs, dim=1)
            output.append(max_pred_index.cpu().numpy()[0].item())
            if max_pred_index == 1 or len(output) >= max_len:
                break
            inputs = self.embed(max_pred_index) 
            inputs = inputs.unsqueeze(1)
        return output
