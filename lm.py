from transformers import BertTokenizer, BertForMaskedLM
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class WordLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')

    def forward(self, sentences: List[str]) -> torch.Tensor:
        """
        :param sentences: List of sentences, must have equal number of words. TODO: auto pad
        :return Probability of words (batch, seq_len)
        """
        input_ids = self.tokenizer(sentences, return_tensors="pt")["input_ids"]
        input_ids = input_ids.to(self.model.device)
        outputs = self.model(input_ids, labels=input_ids)
        scores = outputs[1]
        scores = F.softmax(scores, dim=-1)

        # remove bos and sos
        input_ids = input_ids[:, 1:-1]
        scores = scores[:, 1:-1]

        # batch_size, seq_len, _ = scores.shape
        # output = torch.zeros(batch_size, seq_len).to(scores.device)
        # for i in range(batch_size):
        #     for s in range(seq_len):
        #         output[i, s] = scores[i, s, input_ids[i, s]]
        index = input_ids.unsqueeze(2)
        output = torch.gather(scores, dim=2, index=index)
        return output.squeeze(2)


def test():
    lm = WordLM().cuda()
    scores = lm(['I have the high ground', 'Say hello to my little friend'])
    print(scores)


if __name__ == '__main__':
    test()
