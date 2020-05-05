
import torch
from torchvision import datasets
import numpy as np
import wordcloud
import matplotlib.pyplot as plt
import plotly.express as px
# %matplotlib inline

with open('/data/labels.txt') as f:
    labels= f.read()
with open('/data/reviews.txt') as f:
    reviews=f.read()
print(reviews[:200])
print(labels[:50])

"""## Data Preprocessing"""

from string import punctuation
reviews = reviews.lower() # lowercase, standardize
all_text = ''.join([c for c in reviews if c not in punctuation])
text_split = all_text.split('\n')
all_text = ' '.join(text_split)

# create a list of words
words = all_text.split()

from collections import Counter
counts= Counter(words)
counts.most_common(50)

vocabulary= sorted(counts, key= counts.get, reverse=True)
vocab_to_int= {word: i for i,word in enumerate(vocabulary,1)}
review_ints=[]  
for review in text_split:
    review_ints.append([vocab_to_int[word] for word in review.split()])

print('Unique words: ', len((vocab_to_int))) 
print('Tokenized review: \n', review_ints[:1])

label= labels.split('\n')
label[0]

pos_word=[]
neg_word=[]
for i in range(len(text_split)):
    if label[i]== 'positive':
        for word in text_split[i].split(" "):
            if word=='':
                pass
            else:
                pos_word.append(word)
    else:
        for word in text_split[i].split(" "):
            if word=='':
                pass
            else:    
                neg_word.append(word)

def nonan(x):
    if type(x) == str:
        return x.replace("\n", "")
    else:
        return ""

text = ' '.join([nonan(abstract) for abstract in pos_word])
wc = wordcloud.WordCloud(max_font_size=None, background_color='black', collocations=False,
                      width=1200, height=1000).generate(text)

plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wc) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.title('Words in Good reviews')
plt.show()

text = ' '.join([nonan(abstract) for abstract in neg_word])
wc = wordcloud.WordCloud(max_font_size=None, background_color='black', collocations=False,
                      width=1200, height=1000).generate(text)

plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wc) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.title('Words in Bad reviews')
plt.show()

encoded_labels = np.array([1 if lab == 'positive' else 0 for lab in label])

encoded_labels[:10]

review_lens = Counter([len(x) for x in review_ints])
print("Zero-length reviews: {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))

print('Number of reviews before removing outliers: ', len(review_ints))
non_zero_idx = [ii for ii, review in enumerate(review_ints) if len(review) != 0]
review_ints = [review_ints[ii] for ii in non_zero_idx]
encoded_labels = np.array([encoded_labels[ii] for ii in non_zero_idx])

print('Number of reviews after removing outliers: ', len(review_ints))

def pad_features(reviews_ints, seq_length):
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)
    for i, row in enumerate(reviews_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]
    
    return features

seq_length = 200

features = pad_features(review_ints, seq_length=seq_length)
assert len(features)==len(review_ints), "Your features should have as many rows as reviews."
assert len(features[0])==seq_length, "Each feature row should contain seq_length values."
print(features[:30,:10])

"""## Loading Data"""

split_frac = 0.8
## split data into training, validation, and test data (features and labels, x and y)

split_idx = int(len(features)*split_frac)
train_x, remaining_x = features[:split_idx], features[split_idx:]
train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]

test_idx = int(len(remaining_x)*0.5)
val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]
print("Train set: \t\t{}".format(train_x.shape), 
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))

test_y.shape

import torch
from torch.utils.data import TensorDataset, DataLoader
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
batch_size = 50
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

# obtain one batch of training data
dataiter = iter(train_loader)
sample_x, sample_y = dataiter.next()

print('Sample input size: ', sample_x.size()) # batch_size, seq_length
print('Sample input: \n', sample_x)
print()
print('Sample label size: ', sample_y.size()) # batch_size
print('Sample label: \n', sample_y)

train_on_gpu= torch.cuda.is_available()
if train_on_gpu:
    print('GPU available')

"""## LSTM Model"""

import torch.nn as nn
import torch.nn.functional as f
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, output_size, embed_dim, hidden_dim, n_layers, drop_prob=0.5):
        super(SentimentLSTM, self).__init__()
        self.output_size= output_size
        self.n_layers= n_layers
        self.hidden_dim= hidden_dim
        
        self.embed= nn.Embedding(vocab_size, embed_dim)
        self.lstm= nn.LSTM(embed_dim, hidden_dim, num_layers= n_layers, dropout= drop_prob, batch_first=True)
        self.drop= nn.Dropout(0.3)
        self.fc= nn.Linear(hidden_dim, output_size)
        self.sig= nn.Sigmoid()
    def forward(self, x, hidden):
        batch_size=x.size(0)
        x=x.long()
        embed= self.embed(x)
        lstm_out, hidden= self.lstm(embed, hidden)
        lstm_out= lstm_out.contiguous().view(-1, self.hidden_dim)
        out= self.drop(lstm_out)
        out= self.fc(out)
        out= self.sig(out)
        out= out.view(batch_size,-1)
        out= out[:,-1]
        return out, hidden
    def init_hidden(self,batch_size):
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden

vocab_size = len(vocab_to_int)+1 # +1 for the 0 padding + our word tokens
output_size = 1
embedding_dim = 400
hidden_dim = 256
n_layers = 2

net = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

print(net)

lr=0.001

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

"""## Training"""

epochs = 4 
counter = 0
print_every = 100
clip=5
if(train_on_gpu):
    net.cuda()

net.train()
# train for some number of epochs
for e in range(epochs):
    # initialize hidden state
    h = net.init_hidden(batch_size)
    for inputs, labels in train_loader:
        counter += 1

        if(train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()
        h = tuple([each.data for each in h])
        net.zero_grad()
        output, h = net(inputs, h)
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()
        if counter % print_every == 0:
            val_h = net.init_hidden(batch_size)
            val_losses = []
            net.eval()
            for inputs, labels in valid_loader:
                val_h = tuple([each.data for each in val_h])
                if(train_on_gpu):
                    inputs, labels = inputs.cuda(), labels.cuda()

                output, val_h = net(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())
                val_losses.append(val_loss.item())

            net.train()
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))

"""## Testing"""

test_losses = [] # track loss
num_correct = 0
h = net.init_hidden(batch_size)

net.eval()
for inputs, labels in test_loader:
    h = tuple([each.data for each in h])

    if(train_on_gpu):
        inputs, labels = inputs.cuda(), labels.cuda()
    output, h = net(inputs, h)
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())
    pred = torch.round(output.squeeze())  
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)
print("Test loss: {:.3f}".format(np.mean(test_losses)))
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))

"""## Inference on custom reviews"""

from string import punctuation

def tokenize_review(test_review):
    test_review = test_review.lower() # lowercase
    test_text = ''.join([c for c in test_review if c not in punctuation])

    test_words = test_text.split()
    test_ints = []
    test_ints.append([vocab_to_int[word] for word in test_words])

    return test_ints

test_review_neg= 'Worst movie ever seen. the story was too bad and disappointing.'
test_ints = tokenize_review(test_review_neg)
print(test_ints)

seq_length=200
features = pad_features(test_ints, seq_length)

print(features)

feature_tensor = torch.from_numpy(features)
print(feature_tensor.size())

def predict(net, test_review, sequence_length=200):
    
    net.eval()
    test_ints = tokenize_review(test_review)
    seq_length=sequence_length
    features = pad_features(test_ints, seq_length)
    feature_tensor = torch.from_numpy(features)
    
    batch_size = feature_tensor.size(0)
    h = net.init_hidden(batch_size)
    
    if(train_on_gpu):
        feature_tensor = feature_tensor.cuda()
    output, h = net(feature_tensor, h)
    pred = torch.round(output.squeeze()) 
    print('Prediction value, pre-rounding: {:.6f}'.format(output.item()))
    if(pred.item()==1):
        print("Positive review detected!")
    else:
        print("Negative review detected.")

predict(net, test_review_neg, seq_length)

pos_test_review= 'Amazing movie. The sound and music was too good'
predict(net, pos_test_review, seq_length)
