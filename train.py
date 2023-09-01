import json
import os
import pickle
import time
from os.path import join

import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm

from vqaTools.vqa import VQA
from vqaTools.vqaEvaluation.vqaEval import VQAEval


def compute_score_with_logits(logits, labels):
    '''
    use logits to predict answer represented by one hot
    :param logits:
    :param labels:
    :return: answer represented by one hot
    '''
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def train_pretrain_model(base_model, train_loader, eval_loader, num_epochs,
                         output, label2ans, use_cuda, debug=False, train_qbias=False, cp=False):
    '''
    train the pretrain model
    :param base_model: VQA model
    :param train_loader: dataloader initialized by training set
    :param eval_loader: dataloader initialized by test set (val set in VQA v2)
    :param num_epochs:
    :param output: the model weight output path
    :param label2ans: label to answer dictionary
    :param debug: is it debug
    :param train_qbias: is it use to train q bias model
    :param cp: is it use VQACP dataset
    :return:
    '''
    utils.create_dir(output)
    optim = torch.optim.Adamax(base_model.parameters(), lr=0.0005)
    logger = utils.Logger(os.path.join(output, 'log.txt'))

    for epoch in range(num_epochs):
        base_model.train()

        total_loss = 0
        train_score = 0

        t = time.time()

        for i, (v, q, a, b) in tqdm(enumerate(train_loader), ncols=100,
                                    desc="Epoch %d" % (epoch+1), total=len(train_loader)):
            if use_cuda:
                v = Variable(v).cuda()
                q = Variable(q).cuda()
                a = Variable(a).cuda()
                b = Variable(b).cuda()

            if train_qbias:
                v = torch.zeros_like(v)
            # q_bias = q_bias.flip([0, 1])

            pred, loss, _, _ = base_model(v, None, q, a, b)

            if (loss != loss).any():
                raise ValueError("NaN loss")
            loss.backward()
            nn.utils.clip_grad_norm_(base_model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            total_loss += loss.data.item() * v.size(0)
            train_score += batch_score

            if debug:
                break

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)
        run_eval = epoch == num_epochs - 1 or epoch % 2 == 0
        if run_eval:
            # results, vqa_json_results = evaluate(base_model, eval_loader, label2ans)
            results, vqa_json_results, accuracy = evaluate_ci(base_model, eval_loader, label2ans,
                                                              output, cp, "base_model.json")
            # results["epoch"] = epoch+1
            # results["step"] = total_step
            # results["train_loss"] = total_loss
            # results["train_score"] = train_score
            # all_results.append(results)
            # for item_dict in all_results:
            #     for key in item_dict:
            #         if type(item_dict[key]) is torch.Tensor:
            #             item_dict[key] = item_dict[key].cpu().data.item()
            #
            # json.dump(all_results, open(join(output, "results.json"), "w"))
            #
            # json.dump(vqa_json_results, open(join(output, "vqa_results.json"), "w"))
            #
            eval_score = results["score"]
            bound = results["upper_bound"]
            #
            # if cp:
            #     ques_file_path = "data/vqacp_v2_test_questions.json"
            #     ans_file_path = "data/vqacp_v2_test_annotations.json"
            # else:
            #     ques_file_path = "data/v2_OpenEnded_mscoco_val2014_questions.json"
            #     ans_file_path = "data/v2_mscoco_val2014_annotations.json"
            #
            # vqa = VQA(ans_file_path, ques_file_path)
            # vqaRes = vqa.loadRes(join(output, "vqa_results.json"), ques_file_path)
            #
            # vqaEval = VQAEval(vqa, vqaRes, n=2)
            # vqaEval.evaluate()
            #
            # print("\n")
            # print("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
            # print("Per Answer Type Accuracy is the following:")
            # for ansType in vqaEval.accuracy['perAnswerType']:
            #     print("%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
            # print("\n")

        logger.write('epoch %d, time: %.2f' % (epoch+1, time.time()-t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))

        if run_eval:
            logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

    if train_qbias:
        model_path = os.path.join(output, f'model_epoch{epoch}_qonly.pth')
    else:
        model_path = os.path.join(output, f'model_epoch{epoch}_bmup.pth')
    torch.save(base_model.state_dict(), model_path)

def evaluate(model, dataloader, label2ans):
    score = 0
    upper_bound = 0
    num_data = 0

    all_logits = []
    all_bias = []
    eval_ans_list = []
    model.eval()
    for v, q, a, b in tqdm(dataloader, ncols=100, total=len(dataloader), desc="eval"):
        with torch.no_grad():
            v = v.cuda()
            q = q.cuda()
            pred, _, _, _ = model(v, None, q, None, None)
            all_logits.append(pred.data.cpu().numpy())

            batch_score = compute_score_with_logits(pred, a.cuda()).sum()
            score += batch_score
            upper_bound += (a.max(1)[0]).sum()
            num_data += pred.size(0)
            all_bias.append(b)
            ans_ids = torch.max(pred, 1)[1].data.cpu().numpy()
            for ans_id in ans_ids:
                eval_ans_list.append(label2ans[ans_id])

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)

    results = dict(
        score=score,
        upper_bound=upper_bound,
    )

    vaq_json_results = []
    for ans, entry in zip(eval_ans_list, dataloader.dataset.entries):
        vaq_json_results.append(dict(answer=ans, question_id=entry["question_id"]))

    return results, vaq_json_results


def train_ci(model_nr, model_r, ci_model, train_loader, eval_loader, num_epochs, output,
             label2ans, use_cuda, debug=False, cp=True):
    utils.create_dir(output)
    optim = torch.optim.Adamax(ci_model.parameters(), lr=0.001)
    logger = utils.Logger(os.path.join(output, 'log.txt'))

    total_step = 0
    model_nr.eval()
    model_r.eval()
    loader_len = len(train_loader)
    acc_list = []
    for epoch in range(num_epochs):
        ci_model.train()

        total_loss = 0
        train_score = 0

        t = time.time()
        current_lr = optim.param_groups[0]['lr']
        for i, (v, q, a, b) in enumerate(train_loader):
            total_step += 1

            if use_cuda:
                v = v.cuda()
                q = q.cuda()
                a = a.cuda()
                b = b.cuda()

            # use to capture the causal target (teacher_label)
            with torch.no_grad():
                pred_nr, _, _, _ = model_nr(v, None, q, a, b)
                pred_r, _, _, _ = model_r(v, None, q, a, b)
                teacher_label = torch.abs(pred_nr - pred_r)
                logits = torch.max(teacher_label, 1)[1].data  # argmax
                one_hots = torch.zeros(*a.size()).cuda()
                teacher_label = one_hots.scatter(1, logits.view(-1, 1), 1)

            pred, loss, loss_y, loss_q = ci_model(v, None, q, a, b, teacher_label)

            # use to verify whether loss has an NAN value.
            if (loss != loss).any():
                raise ValueError("NaN loss")

            loss.backward()
            nn.utils.clip_grad_norm_(ci_model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            total_loss += loss.data.item() * v.size(0)
            train_score += batch_score

            print(
                f"\rMode: train epoch: {epoch + 1} , step: [{i}/{loader_len}], loss: {loss.cpu().data.item()} , {loss_y.data.item()}, {loss_q.data.item()}, "
                f"lr={current_lr}",
                end='          ')

            if debug:
                break

        print('\n')

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)

        results, vqa_json_results, accuracy = evaluate_ci(ci_model, eval_loader, label2ans,
                                                          output, cp, "vqa_results_lm_kd_ci_abs.json")

        logger.write('epoch %d, time: %.2f' % (epoch + 1, time.time() - t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))
        acc_list.append(accuracy)

    model_path = os.path.join(output, f'cikd_model_epoch{epoch}_weight.pth')
    torch.save(ci_model.state_dict(), model_path)


def evaluate_ci(ci_model, dataloader, label2ans, output, cp, json_name):
    score = 0
    upper_bound = 0
    num_data = 0

    all_logits = []
    all_bias = []
    eval_ans_list = []
    ci_model.eval()
    for v, q, a, b in tqdm(dataloader, ncols=100, total=len(dataloader), desc="eval"):
        with torch.no_grad():
            v = v.cuda()
            q = q.cuda()

            pred, _, _, _ = ci_model.inference(v, None, q, None, None)
            all_logits.append(pred.data.cpu().numpy())

            batch_score = compute_score_with_logits(pred, a.cuda()).sum()
            score += batch_score
            upper_bound += (a.max(1)[0]).sum()
            num_data += pred.size(0)
            all_bias.append(b)
            ans_ids = torch.max(pred, 1)[1].data.cpu().numpy()
            for ans_id in ans_ids:
                eval_ans_list.append(label2ans[ans_id])

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)

    results = dict(
        score=score,
        upper_bound=upper_bound,
    )

    vqa_json_results = []
    for ans, entry in zip(eval_ans_list, dataloader.dataset.entries):
        vqa_json_results.append(dict(answer=ans, question_id=entry["question_id"]))
    vqa_result_json_name = json_name
    json.dump(vqa_json_results, open(join(output, vqa_result_json_name), "w"))

    if cp:
        ques_file_path = "data/vqacp_v2_test_questions.json"
        ans_file_path = "data/vqacp_v2_test_annotations.json"
    else:
        ques_file_path = "data/v2_OpenEnded_mscoco_val2014_questions.json"
        ans_file_path = "data/v2_mscoco_val2014_annotations.json"

    vqa = VQA(ans_file_path, ques_file_path)
    vqaRes = vqa.loadRes(join(output, vqa_result_json_name), ques_file_path)

    vqaEval = VQAEval(vqa, vqaRes, n=2)
    vqaEval.evaluate()

    print("\n")
    print("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
    print("Per Answer Type Accuracy is the following:")
    for ansType in vqaEval.accuracy['perAnswerType']:
        print("%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
    print("\n")

    return results, vqa_json_results, vqaEval.accuracy
