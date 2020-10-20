from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
from transformers.modeling_t5 import load_tf_weights_in_t5

base_model = "t5-small"
path = "/Users/danielk/ideaProjects/distilled-3b-to-small-8/best_tfmr/"
model = T5ForConditionalGeneration.from_pretrained(path)
tokenizer = T5Tokenizer.from_pretrained(path)
# tokenizer = T5Tokenizer.from_pretrained(base_model)
# model = T5ForConditionalGeneration.from_pretrained('t5-small', return_dict=True)
# model.load_state_dict(distilledmodel)
model.eval()

def run_model(input_string, **generator_args):
    input_string += "</s>"
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    tokens = [tokenizer.decode(x) for x in res]
    print(tokens)


run_model("how many states does the US has? ")
run_model("who is the US president?")
run_model("who got the first nobel prize in physics?")
run_model("when is the next deadpool movie being released?")
run_model("which mode is used for short wave broadcast service?")
run_model("the south west wind blows across nigeria between?")
run_model("what does hp mean in war and order?")
run_model("who wrote the first declaration of human rights?")
run_model("who is the owner of reading football club?")

