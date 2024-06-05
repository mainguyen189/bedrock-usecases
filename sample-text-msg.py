import boto3
from langchain.llms.bedrock import Bedrock

# create the bedrock client
boto3_client = boto3.client('bedrock-runtime')

# set up model inference parameter
inference_modifier = {
    'temperature' : 0.5, # a bit creative
    'top_p' : 1,
    "max_tokens_to_sample" : 9000
}

# create the llm
llm = Bedrock(
    model_id="anthropic.claude-instant-v1",
    client=boto3_client,
    model_kwargs= inference_modifier
)

# generate the response
response = llm.invoke("""
            Human: Write a text message from a single mother and pop idol, Ai Hoshino, telling her babies' father, Hikaru Kamiki, about their children

            Answer:""")

# display answer
print(response)