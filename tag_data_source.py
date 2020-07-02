import argparse

from config import Config
from nltk.tokenize import sent_tokenize
from predictors.svm_predictor import SVMPredictor
from tqdm import tqdm


def tag_dialogues(tagger, dialogues, output_path):
    with open(output_path, "w") as tagger_output_file:

        dialog_das = []
        prev_num_turns = 0
        for dialog in tqdm(dialogues):
            turns = dialog.split("_eos")

            num_turns = len(turns) - 1

            if num_turns < prev_num_turns:
                dialog_das = []


            turn_das = []
            turn = turns[num_turns - 1]

            for sent in sent_tokenize(turn):
                turn_das += [da["communicative_function"] for da in tagger.dialogue_act_tag(sent)]
            dialog_das.append(" ".join(turn_das))

            output = " _eos ".join(dialog_das)
            print(output)
            tagger_output_file.write(output + "\n")
            prev_num_turns = num_turns



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src',
                      type=str,
                      required=True,
                      help="Source file to annotate")
    parser.add_argument("--dst",
                      type=str,
                      required=True,
                      help="Destination file to save annotation results")
    parser.add_argument('--model_path',
                        type=str,
                        default="models/Model.SVM/meta.json",
                        help="Path to SVM tagger model")
    args = parser.parse_args()
    cfg = Config.from_json(args.model_path)


    tagger = SVMPredictor(cfg)

    with open(args.src, "r") as valid_file:
        dialogues = [line.strip() for line in valid_file]

    tagger_outputs = tag_dialogues(tagger, dialogues, args.dst)