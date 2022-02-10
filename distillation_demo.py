from transformer.modeling import TinyBertForSequenceClassification
from transformer.optimization import BertAdam
import argparse
import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model",
                        default=None,
                        type=str,
                        help="The teacher model dir.")
    parser.add_argument("--student_model",
                        default=None,
                        type=str,
                        required=True,
                        help="The student model dir.")
    parser.add_argument("--max_seq_length",
                        default=32,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--num_steps",
                        default=100,
                        type=int,
                        help="Number of training steps to perform using random input")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--weight_decay', '--wd',
                        default=1e-4,
                        type=float,
                        metavar='W',
                        help='weight decay')

    args = parser.parse_args()

    # initialize models
    teacher_model = TinyBertForSequenceClassification.from_scratch(args.teacher_model, num_labels=2, is_student=False)
    student_model = TinyBertForSequenceClassification.from_scratch(args.student_model, num_labels=2, is_student=True,
                                                                   fit_size=teacher_model.hidden_size)

    vocab_size = student_model.bert.embeddings.word_embeddings.num_embeddings

    # initialize Loss and Optimizer
    optimizer = BertAdam(student_model.parameters(),
                         schedule="warmup_linear",
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=args.num_steps,
                         weight_decay=args.weight_decay)
    loss_fct = MSELoss()

    # initialize random data
    train_data = torch.randint(0, vocab_size, (args.num_steps, args.max_seq_length))
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=1)

    for step, input_ids in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True)):

        att_loss = 0.
        rep_loss = 0.

        _, student_atts, student_reps = student_model(input_ids)

        with torch.no_grad():
            _, teacher_atts, teacher_reps = teacher_model(input_ids)

        teacher_layer_num = len(teacher_atts)
        student_layer_num = len(student_atts)
        assert teacher_layer_num % student_layer_num == 0
        layers_per_block = int(teacher_layer_num / student_layer_num)
        new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                            for i in range(student_layer_num)]

        for student_att, teacher_att in zip(student_atts, new_teacher_atts):
            att_loss += loss_fct(student_att, teacher_att)

        new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
        new_student_reps = student_reps
        for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
            rep_loss += loss_fct(student_rep, teacher_rep)

        loss = rep_loss + att_loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

if __name__ == '__main__':
    main()