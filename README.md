# RoBERTa-for-NER

Solving of NER task on the famous dataset <a href='https://paperswithcode.com/dataset/conll-2003'>CONLL2003</a> using RoBERTa architecture and 🤗 Hugging face tools.

### Let's take a look at example from dataset.

`EU    rejects German call to boycott British lamb . `
<br>
`B-ORG O       B-MISC O    O  O       B-MISC  O    O `

Dataset includes following tags: persons (**X-PER**), locations (**X-MISC**), organizations (**X-ORG**) and names of miscellaneous entities (**O**).

### Models training. 

For training RoBERTa was chosen. Also, to compare results of other similar architectures DistillBERT and BERT were added. 

For training of all models Adam optimizer with learning rate 2e-5 was used. Training lasted 13 epochs or 1650 steps. 

Metric was computed after each training epoch on the whole test dataset with batch size 1.

Now let's take a look at test loss and f1-score of all models.

### Test loss comparison

<img src='https://github.com/IOBananaOI/RoBERTa-for-NER/blob/main/output.png?raw=true'>

### F1-score comparison

<img src='https://github.com/IOBananaOI/RoBERTa-for-NER/blob/main/f1.png?raw=true'>

As a result, RoBERTa showed the best result among all the models.
