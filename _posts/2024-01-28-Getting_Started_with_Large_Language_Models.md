---
title: "Getting Started with Large Language Models"
date: 2024-01-28
---
## Why?
This doc is meant to provide the reader with as quick a way into using a LLM on a 'regular' computer.  This post has been written with my workplace in mind, here we have 16gb Mac M1 machines, so everything that follows in this doc will work on that.
## The difficulty of getting started
 I have recently started to play around with Large Language Models (LLMs). But when I started, just getting started was difficult.  Its a new, constantly changing environment; it seems so fast moving and brittle and the hot-new-thing™ seems to drop daily (if not hourly) it can all be a bit overwhelming.

For me, I did my usual 'research' of reading every page of reddit (/r/locallama) and every video on youtube.  This didn't really help, the language was often impenetrable and when i could understand the words, I found that much of the messaging seemed contradictory (so, internet.)

What did help me was, choosing a way to 'serve' models from my own machine and start playing with them.  I could have chosen to use the APIs provided by the likes of OpenAI, and got great results from their models, but I felt that for understanding models, how they work, and, of course, keeping the costs down (OpenAI do charge for API access, on top of their subscription) that a local installation might be better. 

## The options
There are many: llamacpp, koboldcpp, and front-ends such as oobabooga  - this is where a lot of the confusion arises from, even at this first stage, the options appear overwhelming. And the names are ridiculous.

I recommend [ollama](https://ollama.ai/) This will get us running  quickly - go to the website and follow the installation instructions.  Then run: `ollama run llama2`

This will go download llama2 (first time only) and with any luck you should be able to chat with your local llama. 'Why is the sky blue?' is a pretty common 'hello world' for a lot of people here ;-)

Pretty underwhelming, right?  It is true that these locally run models do lack the power of commercially available offerings like chatGPT, but they can still be hugely useful tools. Also, we can install other models.  First lets see what models we have:
`ollama list`
and you will see your models listed, in my case it looks like this:
```
NAME                    ID              SIZE    MODIFIED
deepseek-coder:latest   140a485970a6    776 MB  6 weeks ago
dolphin-mistral:latest  ecbf896611f5    4.1 GB  2 days ago
llama2:latest           78e26419b446    3.8 GB  5 days ago
mistral:latest          d364aa8d131e    4.1 GB  6 weeks ago
tinyllama:latest        2644915ede35    637 MB  5 days ago
zephyr:latest           bbe38b81adec    4.1 GB  2 days ago
```

So, to download these models we can either `ollama pull mistral-openorca` to download that model, or `ollama run mistral-openorca` to download and run. Ollama stores these models in `.ollama/models/`  

For a full list of models that are available, see here: https://ollama.ai/library but bear in mind that all models may not work on your machine, due to hardware requirements.

## What about this shiny new model that EVERYONE is talking about?
It is also possible to install models that have not been made available by the ollama team. Models can be found on HuggingFace, think of that as the github for models.

There are a couple of caveats: 
1. there needs to be a GGUF version of the model
2. you will need to create a model card
3. It still might not work

Regarding the first point, the community often moves fast to create these versions of new models, it often just requires waiting a couple of hours.  More often, than not, they will be released by _TheBloke_ on hugging face: https://huggingface.co/models?sort=modified&search=thebloke

Regarding point 2 - thats _fairly_ straightforward, you can find instructions here: https://github.com/ollama/ollama/blob/main/docs/import.md

I will make the point, however, that when importing the [tinyllama](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF) model, I found the performance, in terms of response quality, to be awful.  I then realised that the model was in fact available though [ollama](https://ollama.ai/library/tinyllama) , D'oh!, and whatever secret-sauce they did to the model meant that I was getting *much* better results with that.  The experience did teach me quite a bit about creating model-cards and working to get better results from a model, so I do recommend it!
(and no, i never got it working as well as the ollama team did)

As for point 3 - with your macbook, you are in a pretty good place to run a lot of models - but you will soon need to know how to read model names.  Again, beyond the scope of this article, but as a rule of thumb: if it has 7B in its name, that means 7Billion parameters, these can usually be run on your machine.  But you may find you need a _quantized_ version, again beyond the scope of this, but the higher the Q number, the worse the results, but the more likely it is, that it will run.  For this, I really recommend experimenting on a case-by-case basis, try it, if it works, great, if not, try a different version. 

Each model on ollama[ has a 'tags' page](https://ollama.ai/library/mistral/tags) with the different versions you can install.
Similarly, on HuggingFace, a model page will generally[ have a Model Card ](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF)with details of different versions.

## What else can ollama do?
So far we have just looked at the models through the cli chat interface. But if we go to: http://localhost:11434/ we will hopefully see that Ollama is running. So we are in fact running an API, and we can test that with:

```sh
curl http://localhost:11434/api/generate -d '{ "model": "llama2", "prompt": "Why is the sky blue?" }'
```

Wait what!?

:-) What you just saw was the streaming response from the model, followed by the embeddings used.  Embeddings will be key later as you explore LLMs but beyond the scope of this article.

We also saw the streaming result, we can switch off streaming with:
```sh
curl http://localhost:11434/api/generate -d '{ "model": "llama2", "prompt": "Why is the sky blue?", "stream": false }'
```
Now, there is obviously much more the API can do, see here: https://github.com/ollama/ollama/blob/main/docs/api.md

## What about libraries?
At the time of writing, ollama have [just released libraries for Python and JS,](https://ollama.ai/blog/python-javascript-libraries) I haven't had much of a play with these, as I have mainly used the API, but these will, i'm sure be worth exploring.

## What's next?
As said at the top of the document, this is just a starter guide for those who want to play around with LLMs, locally.  After this you may wish to explore further with other areas of LLMs or you may wish to evaluate alternatives to ollama, one reason for this may be that a framework you want to use, does not use ollama, perhaps it uses llama.cpp? You may also find that llama.cpp gives you a finer degree of control, this can be useful, for example,  when defining the memory usage of the model. 

