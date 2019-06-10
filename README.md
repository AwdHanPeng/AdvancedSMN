# AdvancedSMN
The first homework for NLP course

references:

Sequential Matching Network: A New Architecture for Multi-turn Response Selection in Retrieval-based Chatbots. ACL 2017 (Long paper)

Directional Skip-Gram: Explicitly Distinguishing Left and Right Context for Word Embeddings. NAACL 2018 (Short Paper)


In the first paper, we can see that SMN uses two channels called M1 and M2 respectively in the First Layer. One of the difference between M1 and M2 is that M2 used a matrix A to achieve a bilinear product but M1 not. So we consider why not use a same method for M1.

In our experiment, we also used the pre-train embedding provided by Tencent nlp lab. As the result, pre-training can help to train a complex network effectively.

Lastly, as the result of our experiment, do not use both of matries M1 and M2 can get the best performance.

More detail can be seen in the report.pdf

Thanks!
