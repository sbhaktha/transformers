"""Evaluation utilities."""
import re
import string

import unicodedata


def normalize_answer(s):
  """Normalize answer."""
  s = unicodedata.normalize("NFD", s)

  def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)

  def white_space_fix(text):
    return " ".join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
  return normalize_answer(prediction) == normalize_answer(ground_truth)


def regex_match_score(prediction, ground_truth):
  try:
    regex = re.compile(ground_truth,
                       flags=re.IGNORECASE + re.UNICODE + re.MULTILINE)
    return regex.match(prediction) is not None
  except re.error:
    return False


def metric_max_over_ground_truths(metric_fn, prediction,
                                  ground_truths):
  scores_for_ground_truths = []
  for ground_truth in ground_truths:
    score = metric_fn(prediction, ground_truth)
    scores_for_ground_truths.append(score)
  return max(scores_for_ground_truths)


def main():
    with open("/Users/danielk/Desktop/nqopen_csv/test.generations", "r") as pred:
        preds = [line.replace("\n", "") for line in pred.readlines()]
    em_list = []
    rm_list = []
    with open("/Users/danielk/Desktop/nqopen_csv/test.tsv", "r") as f:
        all_lines = list(f.readlines())
        for line, prediction in zip(all_lines, preds):
            line_split = line.split("\t")
            assert len(line_split) == 3
            truths = line_split[1].split("///")

            rm_list.append(metric_max_over_ground_truths(regex_match_score, prediction, truths))
            em_list.append(metric_max_over_ground_truths(exact_match_score, prediction, truths))

    print(sum(rm_list) / len(rm_list))
    print(sum(em_list) / len(em_list))



if __name__=="__main__":
    main()