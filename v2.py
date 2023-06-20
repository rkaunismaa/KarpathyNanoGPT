import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
# max_iters = 3000
# we also increased the iterations when we lowered the learning rate
max_iters = 5000
eval_interval = 300
# learning_rate = 1e-2
# we lowered the learning rate after adding the self attention head to the model
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
# ------------

torch.manual_seed(1337)

# dataFile = "/tf/All/Data/Documents/Github/rkaunismaa/KarpathyNanoGPT/data/shakespeare/input.txt"
dataFile = "data/shakespeare/input.txt"

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open(dataFile, 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self attention """

    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
                             
    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x) # (B, T, C)
        k = self.key(x)   # (B, T, C)
        v = self.value(x) # (B, T, C)
        # compute attention scores ('affinities')
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        # perform the weighted aggregation of the values
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
    

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList( [Head(head_size) for _ in range(num_heads)] )
        # we added this after implement residual connections in Block
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        # return torch.cat([h(x) for h in self.heads], dim=-1) # concatenate over the channel dimension (B, T, C) = -1
        # we commented out the above and did this after we implemented residual connections ...
        out = torch.cat([h(x) for h in self.heads], dim=-1) # same as above ... but then we do ..
        out = self.proj(out)
        return out
    
class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            # nn.Linear(n_embd, n_embd),
            # nn.ReLU(),
            # we added this after we added residual connections ...
            # nn.Linear(n_embd, n_embd)
            # but then we did one small tweak to this to reflect what was done in the Vaswani paper ...
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)

        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of head's we would like to use
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        # x = self.sa(x)
        # x = self.ffwd(x)
        # This is how we implement a residual connection ... simple, right!?
        x = x + self.sa(x)
        x = x + self.ffwd(x)
        return x


# super simple bigram model
class BigramLanguageModel(nn.Module):

    # def __init__(self, vocab_size): 'we don't need to pass in the vocab_size in the constructor because we defined it above and its global
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) ... this needs to change .. to .. 
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # and we are adding a postitional layer so that each position from 0 to block_size -1 will also get its own embedding vector
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # this Head layer was added after we coded it ..
        # self.sa_head = Head(n_embd)
        # this layer was added after coding MultiHeadAttention ...
        # self.sa_heads = MultiHeadAttention(4, n_embd//4) # i.e. 4 heads of 8-dimensional self-attention
        # this layer was added after we coded the FeedForward layer
        # self.ffwd = FeedForward(n_embd)
        # this layer was added after we coded the Block layer
        self.blocks =  nn.Sequential(
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4)
        )
        # ... and to go from token embeddings to logits we will need a linear layer ...
        self.lm_head = nn.Linear(n_embd, vocab_size)


    def forward(self, idx, targets=None):

        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        # logits = self.token_embedding_table(idx) # (B,T,C) ... so now this no longer gives us logits but token embeddings ...
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        # and now we can add the positional embeddings to the token embeddings ...
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        # and we now pass the output x from above into the self attention head ..
        # x = self.sa_head(x) # apply one head of self attention. (B, T, C)
        # we added this after adding in MultiHeadAttention sa_heads ..
        # x = self.sa_heads(x) # apply one head of self-attention (B, T, C)
        # we added this after adding in the FeedForward layer ...
        # x = self.ffwd(x) # (B, T, C)
        # we added this layer after coding the Blocks layer ...
        x = self.blocks(x) # (B, T, C)
        # ... and now that we have added the linear layer above, we can now get the logits by ...
        # logits = self.lm_head(tok_emb) # (B, T, vocab_size)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            # logits, loss = self(idx)
            # focus only on the last time step
            #logits = logits[:, -1, :] # becomes (B, C)
            
            # the above changed after we added Head ...
            # crop ids to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the prediction
            logits, loss = self(idx_cond)
            # focus on only the last time step
            logits = logits[:, -1, :] # becomes (B, C)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# model = BigramLanguageModel(vocab_size) # no longer are passing in the vocab_size
model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
