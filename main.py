import torch.nn as nn
import torch
import torch.utils.data as data
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
import torch.optim as optim
import gc

from dataloader import get_loader
from SeqLSTMAE import Seq2Seq
from utils import init_weights, to_var, epoch_time
import os
import sys
import time
import math


root_path = master_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

EMB_DIM = 1000
HID_DIM = 1000
DROPOUT = 0.2
NUM_LAYERS = 4

BATCH_SIZE = 60 
N_EPOCHS = 10
CLIP = 1


log_step = 20

def main():

    vocab_file = open(root_path + "/data/dictionary", "r")
    docs_file = open(root_path + "/data/train_target_permute_segment.txt", "r")

    lines = docs_file.readlines()
    lines = lines[:600000]#[:int(0.5*len(lines))]
    train_loader, test_loader, vocab_size, PAD_IDX = get_loader(lines, vocab_file.readlines(), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    lines = None

    vocab_file.close()
    docs_file.close()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("***Creating and initializing Model***")
    model = Seq2Seq(vocab_size, emb_dim=EMB_DIM, hid_dim=HID_DIM, n_layers=NUM_LAYERS, dropout=DROPOUT, device=device).to(device)

    model.apply(init_weights)

    print(model)

    params = list(model.parameters())

    optimizer = optim.Adam(params)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)


    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, criterion, CLIP, epoch, N_EPOCHS)
        valid_loss = evaluate(model, test_loader, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut1-model.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')



def train(model, dataloader, optimizer, criterion, clip, epoch, num_epoch):

    model.train()

    epoch_loss = 0

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    total_steps = len(dataloader)

    print("***Starting training...***")

    for i, (src, trg, lengths) in enumerate(dataloader):

        optimizer.zero_grad()

        src = to_var(src)
        trg = to_var(trg)

        output = model(src, lengths)

        # trg = [batch size, trg sent len]
        # output = [batch size, src_sent_len, output dim]

        output = output[:, 1:].contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        # trg = [(trg sent len - 1) * batch size]
        # output = [(trg sent len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        if epoch >= 5:
            scheduler.step()

        optimizer.step()

        gc.collect()
        torch.cuda.empty_cache()

        epoch_loss += loss.detach().item()


        if i % log_step == 0 and i != 0:
            print('Epoch [%d/%d], Step [%d/%d], Partial current loss: %f'
                  % (epoch, num_epoch, i, total_steps, epoch_loss/i))

            print("Memory allocated: "+str(torch.cuda.memory_allocated() / 1024 ** 2)+" MB")


    return epoch_loss / total_steps


def evaluate(model, dataloader, criterion):
    model.eval()

    epoch_loss = 0

    print("***Starting evaluation***")

    for i, (src, trg, lengths) in enumerate(dataloader):

        src = to_var(src)
        trg = to_var(trg)

        output = model(src, lengths, 0)

        # trg = [batch size, trg sent len]
        # output = [batch size, src_sent_len, output dim]

        output = output[:, 1:].contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        # trg = [(trg sent len - 1) * batch size]
        # output = [(trg sent len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

main()