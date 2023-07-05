import os
import json
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union

from transformers.trainer import PredictionOutput
from transformers.tokenization_utils import PreTrainedTokenizer

import evaluate
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")  # TODO: decide which metrics to load


from .peft_trainer import PeftTrainer

from .other import get_logger, IGNORE_INDEX


logger = get_logger(__name__)


@dataclass
class ComputeMetrics:
    r"""
    Wraps the tokenizer into metric functions, used in Seq2SeqPeftTrainer.
    """

    tokenizer: PreTrainedTokenizer

    def __call__(self, eval_preds: Sequence[Union[np.ndarray, Tuple[np.ndarray]]]) -> Dict[str, float]:
        r"""
        Uses the model predictions to compute metrics.
        """
        preds, labels = eval_preds
        # score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
        # NOTE by zian: not tested
        decoded_hyps = []
        decoded_refs = []

        for pred, label in zip(preds, labels):
            pred_pad_len, label_pad_len = np.sum(pred == IGNORE_INDEX), np.sum(label == IGNORE_INDEX)
            pred = pred[len(label) - label_pad_len : len(pred) - pred_pad_len] # remove prompts
            # label = label[:len(label) - label_pad_len]
            label = np.where(label==IGNORE_INDEX,0,label)

            hypothesis = self.tokenizer.decode(pred, skip_special_tokens=True)
            reference = self.tokenizer.decode(label, skip_special_tokens=True)
        
            decoded_hyps.append(hypothesis)
            decoded_refs.append(reference)
        
        rouge_results = rouge.compute(predictions=decoded_hyps, references=decoded_refs)
        bleu_results = bleu.compute(predictions=decoded_hyps, references=[[ref] for ref in decoded_refs])

        score = rouge_results
        score['bleu'] = bleu_results['bleu']

        return score


class Seq2SeqPeftTrainer(PeftTrainer):
    r"""
    Inherits PeftTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def save_predictions(
            self,
            predict_results: PredictionOutput
    ) -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")
        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for pred, label in zip(predict_results.predictions, predict_results.label_ids):
                pred_pad_len, label_pad_len = np.sum(pred == IGNORE_INDEX), np.sum(label == IGNORE_INDEX)
                pred = pred[len(label) - label_pad_len : len(pred) - pred_pad_len] # remove prompts
                # label = label[:len(label) - label_pad_len]
                label = np.where(label==IGNORE_INDEX,0,label)


                pred = self.tokenizer.decode(pred, skip_special_tokens=True)
                label = self.tokenizer.decode(label, skip_special_tokens=True)

                res.append(json.dumps({"label": label, "predict": pred}, ensure_ascii=False, indent=2))

            writer.write("\n".join(res))
