import replicate

# The mistralai/mistral-7b-instruct-v0.2 model can stream output as it's running.
for event in replicate.stream(
    "mistralai/mistral-7b-instruct-v0.2",
    input={"prompt": "how are you doing today?"},
):
    print(str(event), end="")








# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
# import torch
#
# import openai
# import pdfplumber
# import time
#
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
# import torch
#
#
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
# )
#
# # Define the text of the academic AI paper
# paper_text = ""
#
# with pdfplumber.open("segment.pdf") as pdf:
#     for page in pdf.pages:
#         # Extract text from the current page
#         text = page.extract_text()
#         if text:
#             paper_text += text + '\n'  # Add a newline character after each page's text
#
# # split paper into chunks
# chunks = []
# # define chunk size
# chunk_size = 500
# for i in range(0, len(paper_text), chunk_size):
#     chunk = paper_text[i:i+chunk_size]
#     chunks.append(chunk)
#
#
# # Define the prompt/question for the model
# prompt = """
# The following is text from an artificial intelligence research paper. The paper is split into chunks, wait for all the chunks to finish uploading."
# """
#
# # Concatenate the prompt and paper text
# input_text = prompt + paper_text
#
# model_name = "/kaggle/input/mistral/pytorch/7b-v0.1-hf/1"
#
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     load_in_4bit=True,
#     quantization_config=bnb_config,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
#     trust_remote_code=True,
# )
#
# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer = tokenizer,
#     torch_dtype=torch.bfloat16,
#     device_map="auto"
# )
#
# sequences = pipe(
#     prompt,
#     do_sample=True,
#     max_new_tokens=100,
#     temperature=0.7,
#     top_k=50,
#     top_p=0.95,
#     num_return_sequences=1,
# )
#
# model_responses = []
#
# # Iterate through each paper chunk
# counter = 0
# for chunk in chunks:
#     # Initialize the completion object for each chunk
#     txt = "Below is chunk {counter:d} of {num_chunks:d}".format(counter = counter, num_chunks=len(chunks)) + "\n" + chunk
#
#     sequences = pipe(
#         txt,
#         do_sample=True,
#         max_new_tokens=100,
#         temperature=0.7,
#         top_k=50,
#         top_p=0.95,
#         num_return_sequences=1,
#     )
#
#     counter += 1
#
#     # Retrieve and append the completion result
#     completion_result = sequences[0]['generated_text']
#     model_responses.append(completion_result)
#     # Optional: Add a small delay between API calls to avoid rate limits
#     time.sleep(1)
#
# # Process the model responses and analyze the hyperparameters
# # You can perform analysis on model_responses here
#
# # Print or save the results
# for idx, response in enumerate(model_responses):
#     print(f"Analysis for chunk {idx + 1}:")
#     print(response)
#
