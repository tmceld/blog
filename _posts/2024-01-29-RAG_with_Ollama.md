---
title: "Retrieval Augmented Generation (RAG) with Ollama"
date: 2024-01-29
mermaid: true
---

RAG: Retrival Augmented Generation
LLM: Large Language Model
## Before we begin...
This guide, is largely based on this excellent video by [@pixegami](https://www.youtube.com/@pixegami): [RAG + Langchain Python Project Easy AIChat For Your Docs (www.youtube.com)](https://www.youtube.com/watch?v=tcqEUSNCn8I) I have amended the code slightly to use a local LLM and HuggingFace embeddings.

All my code for this can be [found here](https://github.com/tmceld/RAG_with_ollama/)
## Why?
LLM's are fairly limited, but if we can add our own data, be it in the form of pdfs, text files or even databases, we can leverage the LLM to 'talk to us' about our own data
## How?
For this we will assume that we are using ollama as per our previous guide: [Getting Started with Large Language Models](https://toast.weblog.lol/2024/01/getting-started-with-large-language-models) 

We are going to be using [Langchain](https://www.langchain.com/) to implement our RAG.  There are many many other ways to do this, one in particular that looks interesting is[ llama-index](https://github.com/run-llama/llama_index).  But langchain has been around for a while, has widespread adoption and great docs, so we will start with it.
## Setup

Get some data
```
curl "https://www.gutenberg.org/cache/epub/730/pg730.txt" > "data/books/Oliver Twist.txt"
curl "https://www.gutenberg.org/cache/epub/766/pg766.txt" > "data/books/David Copperfield.txt"
curl "https://www.gutenberg.org/cache/epub/1400/pg1400.txt" > "data/books/Great Expectations.txt"
```

## A little bit about embeddings
> Embeddings are vector representations of text that capture their meaning
> in Python, this is a list of numbers. you can think of them as co-ordinates in multi-dimensional space
![](https://cdn.some.pics/toast/65b669be7c697.png)
source: [RAG + Langchain Python Project Easy AIChat For Your Docs - YouTube (www.youtube.com)](https://www.youtube.com/watch?v=tcqEUSNCn8I)

To actually generate a vector from a word, we need an llm, and this is usually just an API or a function.  In  [@pixegami](https://www.youtube.com/@pixegami)'s example on youtube, He uses OpenAI's embeddings, here we are just going to use some from HuggingFace.  By default this uses `sentence-transformers/distilbert-base-nli-mean-tokens`  [source](https://js.langchain.com/docs/integrations/text_embedding/hugging_face_inference) Now, i don't know if those docs are out of date, but this model looks like its [deprecated on the HuggingFace site](https://huggingface.co/sentence-transformers/distilbert-base-nli-mean-tokens) so we will pass in a different model. In this case I followed the link on the deprecated page and took the top model it linked to, but if I was to explore other embedding models, i would probably start here: https://huggingface.co/sentence-transformers


here is an example: (Full code [here](https://github.com/tmceld/RAG_with_ollama/blob/main/compare_embeddings.py))
```python
from langchain.evaluation import load_evaluator
from langchain_community.embeddings import HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings()
hf_evaluator = load_evaluator("pairwise_embedding_distance", embeddings=embedding_model)
x = hf_evaluator.evaluate_string_pairs(prediction="apple", prediction_b="orange")
print(x)
```
`Comparing (apple, orange): {'score': 0.4041740589510623}`

So, the similarity of Apple & Orange is 0.40...  is that a good score?  what does that mean?
let try things that are not related:

```python
words = ("apple", "truck")
x = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=words[1])
```
`Comparing (apple, Truck): {'score': 0.4909352601770862}`

```python
words = ("apple", "suitcase")
x = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=words[1])
```
`Comparing (apple, suitcase): {'score': 0.44280123239920544}`

```python
words = ("apple", "orchard")
x = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=words[1])
```
`Comparing (apple, orchard): {'score': 0.35052391014362416}`
OK, so it looks like the more closely related a word is, the lower the score.  Lets try one more

```python
words = ("apple", "iphone")
x = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=words[1])
```
`Comparing (apple, iphone): {'score': 0.2574609705656187}`

This is interesting, it tells us that in this embedding model, there is probably more mentions of the word Apple, relating to the company, than apple, meaning the fruit.

## Building our RAG
The full code for this section can be [found here](https://github.com/tmceld/RAG_with_ollama/blob/main/documents2vectorDB.py)
So to work with our text documents we need to load it in, split it into chunks and then embed it.
### Loading the data
```python
from langchain_community.document_loaders import DirectoryLoader

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.txt")
    documents = loader.load()
    return documents
```

This is pretty straightforward. For more on Document loaders [see here](https://python.langchain.com/docs/modules/data_connection/document_loaders/).  Also of interest may be [webBaseLoader](https://python.langchain.com/docs/integrations/document_loaders/web_base)which will fetch web pages. And [Wikipedia](https://python.langchain.com/docs/integrations/document_loaders/wikipedia)which will search and return wikipedia articles.

The one thing worth pointing out here is that this will return a Document, but it also returns metadata _about_ the document, and, whats more, we can add to that metadata.
### Splitting the text
Next we work on splitting the documents.  This is handled by the following:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                    chunk_overlap=100,
                                                    length_function=len,
                                                    add_start_index=True
    )
    documents = load_documents()
    chunks = text_splitter.split_documents(documents)
    print(f"{len(documents)} documents split into {len(chunks)} chunks")
    document = chunks[47]
    print(document.page_content)
    print(document.metadata)
    return chunks
```

Here we are using [RecursiveCharacterTextSplitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter) for other types of splitters that may perform better for your particular case, see here:  https://python.langchain.com/docs/modules/data_connection/document_transformers/
### Loading into DB
Next we want to store the embeddings in a database.  In this case we will use [Chromadb](https://www.trychroma.com/) many guides use [pinecone](https://www.pinecone.io/) and other options are available.

```python
from langchain.vectorstores.chroma import Chroma
CHROMA_PATH = "chroma"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

def save_to_chroma(chunks):
    # set up a db for the chunks
    if not os.path.exists(CHROMA_DB_PATH):
        os.mkdir(CHROMA_DB_PATH)
    print("making db")
    # create new db from documents
    db = Chroma.from_documents(
        documents=chunks,
        embedding=HuggingFaceEmbeddings(model_name=EMBED_MODEL), 
        
        persist_directory=CHROMA_DB_PATH
    )
    print('peristing')
    db.persist()
    print('persisted')
    print(f"Saved {len(chunks)} chunks into {CHROMA_DB_PATH}")
```

Now, be warned, this can take a while.  On my 16gb Mac M1, this took about 20 minutes for 4 Dickens books.  I have persisted the database, so the sensible thing to do here would be to compare whats in the database with what is in the Documents, but for demonstration I have not done that yet.

Also note, that if you change the embeddings model that you use, you may wish to clear out the old db first:
```python
if os.path.exists(CHROMA_PATH):
	shutil.rmtree(CHROMA_PATH)
```

## Query for the data
Now we have our embeddings in the vector database we are ready to query it with a LLM, all we need are our database and the same embedding function that we used to create it.

What we are doing here is, taking our query: 'What people did Oliver Twist meet?' and use the same function as was used on the db, to turn that into an embedding, then scan through the DB and find N number of snippets that have a similar score to our question. From that the LLM reads through those chunks and decides what response to give to the user.

So, first we will load the ChromaDB from earlier:

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
CHROMA_PATH = "chroma"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
# Prepare the DB.
embedding_function = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
```

Once that has loaded, next we search the chunks for snippets with a low score:
```python
# Search the DB.
results = db.similarity_search_with_relevance_scores(query_text, k=8)
```

here we are passing in out query text ('Who was Oliver Twist?') and telling it how many snippets we want (8).  This will return us a tuple containing the chunk and a float representing its 'relevance score'

We can also put in a check here, if there are no results, or the relevance of the first is less than a certain value, then tell the user that nothing was found
```python
if len(results) == 0 or results[0][1] < 0.7:
	print(f"Unable to find matching results.")
	return
```

### Crafting the response
The full code for this can be[ found here](https://github.com/tmceld/RAG_with_ollama/blob/main/query_data.py).

First thing we need is a prompt template, in our case it looks like this:

```python
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
```

This takes in 2 pieces of information:
1. the context, this is the chunks that we got from the db
2. the actual query itself.

so first we will build up the `{context}` for the prompt:
```python
from langchain.prompts import ChatPromptTemplate
context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(context=context_text, question=query_text)
```
So, this joins together all of the chunks that were found, and creates a string that also contains the query that we, the user, ask.

Lastly, we send all of that to the LLM:
```python
LLM='mistral'
model = ChatOllama(model=LLM, temperature=0.1)
response_text = model.invoke(prompt)
```

Finally, its useful so show the metadata to give us an idea of what sources were used:
```python
sources = [doc.metadata.get("source", None) for doc, _score in results]
formatted_response = f"Response: {response_text}\nSources: {sources}"
print(formatted_response)
```


## What's Next?
If you have had that up and running, you will find that, while it works, and in some cases works very well, it has some idiosyncrasies that would need to be ironed out.

There is a lot of scope for tweaking with models used, embedding models used, the score filters and the K value for chunks returned.

I found, in a lot of cases I would only get sources from one document - but if i had a question over all the texts, more often than not, it would be based only on the first document it hit.

So, a good starting point - but lots to do!
