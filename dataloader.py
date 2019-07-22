
import torch
import torch.utils.data as data
import torch.nn.utils.rnn as rnn



class TextDataset(data.Dataset):
    def __init__(self, lines, vocabulary_lines):

        print("***Initializing Dataset***")
        self.word_to_ix, self.vocab_size, self.ix_to_word = self.init_w2ix(vocabulary_lines)
        print("***Vocabulary length:%d" %(self.vocab_size))
        self.word_tensors = torch.tensor([self.word_to_ix[w] for w in self.word_to_ix], dtype = torch.long)
        self.src, self.trg = self.define_seq(lines)


    def init_w2ix(self, vocab_lines):
        word_to_ix = {word.split("\n")[0]: i+1 for i, word in enumerate(vocab_lines)}
        word_to_ix["<PAD>"] = 0
        word_to_ix["<EOS>"] = len(vocab_lines) + 1
        word_to_ix["<SOS>"] = len(vocab_lines) + 2
        vocab_size = len(word_to_ix)

        ix_to_word = {ix: word for word, ix in word_to_ix.items()}

        return word_to_ix, vocab_size, ix_to_word

    def define_seq(self, lines):
        srcs = []
        trgs = []
        for line in lines:
            if line != "\n":
                src = line.replace("\n", "").split()
                if len(src)>=10 and len(src)<=15:
                    trg = src[::-1]
                    src.insert(0, self.word_to_ix["<SOS>"])
                    trg.insert(0, self.word_to_ix["<SOS>"])
                    src.insert(len(line), self.word_to_ix["<EOS>"])
                    trg.insert(len(line), self.word_to_ix["<EOS>"])
                    src = [int(w) for w in src]
                    trg = [int(w) for w in trg]

                    srcs.append(src)
                    trgs.append(trg)

        return srcs, trgs


    def get_vocab_size(self):
        return self.vocab_size

    def get_pad_idx(self):
        return self.word_to_ix["<PAD>"]


    def __getitem__(self, index):
        src = torch.tensor(self.src[index])
        trg = torch.tensor(self.trg[index])
        return (src, trg)


    def __len__(self):
        return len(self.src)


def pad_batch(batch):
    sorted_batch= sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    src = [x[0] for x in sorted_batch]
    trg = [x[1] for x in sorted_batch]
    src_padded = torch.nn.utils.rnn.pad_sequence(src, batch_first=True)
    trg_padded = torch.nn.utils.rnn.pad_sequence(trg, batch_first=True)
    lengths = torch.LongTensor([len(x) for x in src])

    return (src_padded, trg_padded, lengths)


def get_loader(lines, vocabulary_lines, batch_size, shuffle, num_workers):
    dataset = TextDataset(lines=lines, vocabulary_lines=vocabulary_lines)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    print("***Splitting dataset for train and val***")
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    print("***TRAIN_SET LENGTH: [%d], VAL_SET LENGTH: [%d]" %(len(train_dataset), len(test_dataset)))
    vocab_size = dataset.get_vocab_size()
    pad_idx = dataset.get_pad_idx()


    return torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=pad_batch), torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                collate_fn=pad_batch), vocab_size, pad_idx


def get_TEST_loader(lines, vocabulary_lines, batch_size, shuffle, num_workers):
    dataset = TextDataset(lines=lines, vocabulary_lines=vocabulary_lines)
    vocab_size = dataset.get_vocab_size()
    pad_idx = dataset.get_pad_idx()

    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=pad_batch), vocab_size, pad_idx