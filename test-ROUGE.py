import torch.nn as nn
import torch
import torch.utils.data as data
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
import torch.optim as optim
import gc

from dataloader import get_TEST_loader
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

BATCH_SIZE = 32
N_EPOCHS = 10
CLIP = 1


log_step = 20

def main():

    vocab_file = open(root_path + "/data/dictionary", "r")
    docs_file = open(root_path + "/data/train_target_permute_segment.txt", "r")

    lines = docs_file.readlines()
    lines = lines[len(lines) - 500000:]#[:int(0.5*len(lines))]
    test_loader, vocab_size, PAD_IDX = get_TEST_loader(lines, vocab_file.readlines(), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    lines = None

    vocab_file.close()
    docs_file.close()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("***Creating and initializing Model***")
    model = Seq2Seq(vocab_size, emb_dim=EMB_DIM, hid_dim=HID_DIM, n_layers=NUM_LAYERS, dropout=DROPOUT, device=device).to(device)

    model.load_state_dict(torch.load("tut1-model.pt"))

    model.eval()

    softmax = nn.LogSoftmax(2)

    results = open("results/results.txt", "a")
    trgs = open("results/targets.txt", "a")
    with torch.no_grad():
        for i, (src, trg, lengths) in enumerate(test_loader):
            src = to_var(src)
            trg = to_var(trg)

            output = model(src, lengths, 0)

            # trg = [batch size, trg sent len]
            # output = [batch size, src_sent_len, output dim]

            output = softmax(output)
            final = torch.zeros(output.shape[0], output.shape[1])


            for i in range(0, output.shape[1]):
                final[:,i] = output[:,i].max(1)[1]

            result, trg = ix2word(final, trg, test_loader.dataset.ix_to_word, results, trgs)





def ix2word(output, trg, ix2word, results, trgs):
    output = output.cpu().numpy()
    trg = trg.cpu().numpy()
    curr_out = []
    curr_trg = []

    for i, row in enumerate(output.tolist()):
        out_sent = ""
        out_trg = ""
        for j, ix in enumerate(row):
            out_sent = out_sent + ix2word[row[j]] + " "
            out_trg = out_trg + ix2word[trg[i][j]] + " "
        out_sent = out_sent + "\n"
        out_trg = out_trg + "\n"
        curr_out.append(out_sent)
        curr_trg.append(out_trg)
    results.writelines(curr_out)
    trgs.writelines(curr_trg)
    return curr_out, curr_trg




main()