import torch
from torch.utils.data import DataLoader, Dataset  
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn as nn
import torch.optim as optim

# Define Constants
ft_epochs = 10
input_dim = 768 # GPT2 hidden size
hidden_dim = 1024
lambda_coef = 0.001

# Load GPT-2 Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Define the Encoder and Decoder
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.b_e = torch.nn.Parameter(torch.zeros(hidden_dim)).to(device)
        self.b_d = torch.nn.Parameter(torch.zeros(input_dim)).to(device)

    def forward(self, x):
        x -= self.b_d
        f = torch.relu(self.linear(x) + self.b_e)
        return f

class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.b_d = torch.nn.Parameter(torch.zeros(output_dim)).to(device)

    def forward(self, f):
        x_hat = self.linear(f) + self.b_d
        return x_hat

# Define custom dataset
class GPT2Dataset(Dataset):
    def __init__(self, sentences, tokenizer, max_length=512):
        self.encodings = tokenizer(sentences, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = item['input_ids'].clone()
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

# Assume we have a list of sentences, replace this with your own sentences list
sentences = ["Hello, world!"] * 1000 
dataset = GPT2Dataset(sentences, tokenizer)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

encoder = Encoder(input_dim, hidden_dim).to(device)
decoder = Decoder(hidden_dim, input_dim).to(device)

# Define loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

# Train the Autoencoder
for epoch in range(ft_epochs):
    for i, batch in enumerate(dataloader):
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(inputs, attention_mask=attention_mask)
            features = outputs.last_hidden_state

        f_hat = encoder(features)
        x_hat = decoder(f_hat)

        loss = loss_fn(x_hat, features) + lambda_coef * torch.norm(f_hat, 1)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f'Epoch [{epoch+1}/{ft_epochs}], Loss: {loss.item():.4f}')

# Freeze all GPT2 parameters and unfreeze only the last layer
for name, param in model.named_parameters():
    param.requires_grad = False
    if 'lm_head' in name:
        param.requires_grad = True

# Fine-tune last layer of GPT2
ft_optimizer = optim.Adam(model.parameters(), lr=0.001)
ft_loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

for epoch in range(ft_epochs):
    for i, batch in enumerate(dataloader):
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            features = encoder(model.transformer(inputs).last_hidden_state)
        
        logits = model.lm_head(features)
        loss = ft_loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        loss.backward()
        ft_optimizer.step()
        ft_optimizer.zero_grad()

    print(f'Epoch [{epoch+1}/{ft_epochs}], Loss: {loss.item():.4f}')

torch.save(model.state_dict(), 'gpt2_finetuned.pt')