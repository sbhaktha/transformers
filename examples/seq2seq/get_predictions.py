from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration

if True:
    base_model = "t5-3b"
    path = "/Users/danielk/ideaProjects/natural_questions_open_light_mixture/3B/pytorch_model"
    model = T5ForConditionalGeneration.from_pretrained(path)
    tokenizer = T5Tokenizer.from_pretrained(path)
    model.eval()
else:
    from transformers.modeling_t5 import load_tf_weights_in_t5
    base_model = "t5-small"
    model = T5ForConditionalGeneration.from_pretrained(base_model)
    tokenizer = T5Tokenizer.from_pretrained(base_model)
    # model.save_pretrained("/Users/danielk/Desktop/t5-large")
    # tokenizer.save_pretrained("/Users/danielk/Desktop/t5-large")

    load_tf_weights_in_t5(model, None, "/Users/danielk/ideaProjects/small_standard/")
    model.eval()

def run_model(input_string, **generator_args):
    # input_string += "</s>"
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    tokens = [tokenizer.decode(x) for x in res]
    print(tokens)


run_model("how many states does the US has? ")
run_model("how many states does the US has? ", num_return_sequences=1)
run_model("who is the US president?")
run_model("who is the US president?", num_return_sequences=1)
run_model("who got the first nobel prize in physics?")
run_model("who got the first nobel prize in physics?", num_return_sequences=1)
run_model("when is the next deadpool movie being released?")
run_model("when is the next deadpool movie being released?", num_return_sequences=1)
run_model("which mode is used for short wave broadcast service?")
run_model("which mode is used for short wave broadcast service?", num_return_sequences=1)
run_model("the south west wind blows across nigeria between?")
run_model("the south west wind blows across nigeria between?", num_return_sequences=1)
# run_model("what does hp mean in war and order?")
# run_model("who wrote the first declaration of human rights?")
# run_model("who is the owner of reading football club?")

