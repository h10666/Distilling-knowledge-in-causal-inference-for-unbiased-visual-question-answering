import torch
import torch.nn as nn
from attention import NewAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet
from vqa_debias_loss_functions import LearnedMixin


class BaseModel_BCELoss(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier, use_sigmoid=True):
        super(BaseModel_BCELoss, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.bceloss = nn.BCELoss(reduction='sum').cuda()
        self.bias_lin = torch.nn.Linear(1024, 1)
        self.use_sigmoid = use_sigmoid

    def forward(self, v, _, q, labels, bias):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)  # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1)  # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        if self.use_sigmoid:
            logits = torch.sigmoid(logits)

        if labels is not None and self.use_sigmoid:
            loss = self.bceloss(logits, labels)
        else:
            loss = None
        return logits, loss, joint_repr, att
        # return logits, loss, joint_repr

    def inference(self, v, _, q, labels, teacher_labels):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)  # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1)  # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits, None, None, att


class BaseModel_CI(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier):
        super(BaseModel_CI, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.bceloss = nn.BCELoss(reduction='sum').cuda()
        self.kl = nn.KLDivLoss(reduction='sum')
        # self.bias_scale = torch.nn.Parameter(torch.from_numpy(np.ones((1, ), dtype=np.float32)*1.2))
        self.bias_lin = torch.nn.Linear(1024, 1)

    def forward(self, v, _, q, labels, bias, return_weights=False):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)  # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1)  # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        logits = torch.sigmoid(logits)

        if labels is not None:
            labels_sum = labels.sum(1)
            logits_sum = logits.sum(1)
            kl_v = self.kl(logits_sum, labels_sum)
            loss = self.bceloss(logits, labels) - kl_v
        else:
            loss = None
        return logits, loss, joint_repr


class CI_Model(nn.Module):
    def __init__(self, alpha, student_model: BaseModel_BCELoss):
        super(CI_Model, self).__init__()
        self.student_model = student_model
        self.mask_linear = nn.Linear(1024, 1)
        self.learned_mix_1 = LearnedMixin(w=0.36)
        self.learned_mix_2 = LearnedMixin(w=0.36)
        # self.learned_mix_1 = LearnedMixinWithTL(w=0.36)
        # self.learned_mix_2 = LearnedMixinWithTL(w=0.36)
        self.bceloss = nn.BCELoss(reduction='sum').cuda()
        self.alpha = alpha

    def forward(self, v, _, q, labels, b, teacher_labels):
        logits_y, _, joint_repr_y, _ = self.student_model(v, _, q, labels, None)
        # logits_q, _, joint_repr_q = self.student_model(v, _, q, teacher_labels, None)

        if labels is not None:
            loss_y = self.learned_mix_1(joint_repr_y, logits_y, b, labels)
            loss_q = self.learned_mix_2(joint_repr_y, logits_y, b, teacher_labels)
            loss = loss_y + loss_q * self.alpha
        else:
            loss = None
            loss_y = None
            loss_q = None
        return logits_y, loss, loss_y, loss_q

    def inference(self, v, _, q, labels, teacher_labels):
        logits_y, _, _, att = self.student_model(v, _, q, None, None)

        return logits_y, None, None, att

class Simple_KD_CI_Model(nn.Module):
    def __init__(self, alpha, student_model: BaseModel_BCELoss):
        super(Simple_KD_CI_Model, self).__init__()
        self.student_model = student_model
        self.alpha = alpha

    def forward(self, v, _, q, labels, b, teacher_labels):
        logits_y, loss_y, _ = self.student_model(v, _, q, labels, None)
        _, loss_q, _ = self.student_model(v, _, q, teacher_labels, None)

        if labels is not None:
            loss = loss_y + loss_q * self.alpha
        else:
            loss = None
        return logits_y, loss, None, loss_y, loss_q

    def inference(self, v, _, q, labels, teacher_labels):
        logits_y, _, _, att = self.student_model(v, _, q, None, None)

        return logits_y, None, None, att

def build_model_bceloss(dataset, num_hid, use_sigmoid=True):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    return BaseModel_BCELoss(w_emb, q_emb, v_att, q_net, v_net, classifier, use_sigmoid)