from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score

def compute_bleu(predictions, references):
    smooth = SmoothingFunction().method1
    bleu1, bleu4 = 0, 0
    for pred, ref in zip(predictions, references):
        bleu1 += sentence_bleu([ref.split()], pred.split(),
                                weights=(1, 0, 0, 0),
                                smoothing_function=smooth)
        bleu4 += sentence_bleu([ref.split()], pred.split(),
                                weights=(0.25,) * 4,
                                smoothing_function=smooth)
    return bleu1 / len(predictions), bleu4 / len(predictions)

def compute_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
    )
    scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref, pred)
        for key in scores:
            scores[key] += s[key].fmeasure
    return {k: v / len(predictions) for k, v in scores.items()}

def compute_bertscore(predictions, references):
    P, R, F1 = bert_score(predictions, references, lang="en")
    return P.mean().item(), R.mean().item(), F1.mean().item()

