# Text-Generation

This project trains a language model on Shakespeare and uses it to generate text. The model will be bare-bones but has the same underlying ideas as much more advanced models that are in wide use. Simply put, language models assign a likelihood (probability) to a given sentence. For example, the sentence "dog is barking" has a higher likelihood than "bark dog is", which is grammatically incoherent.

### <font color='blue'>Bigram and trigram models</font>

Suppose we have a sentence composed of words (or tokens) $x_1, x_2, ..., x_n$. We want to be able to compute the probability of this sequence, $P(x_1, x_2, ... x_n)$. The distribution can always be factored as

$$  P(x_1, x_2, ... x_n)  =  P(x_1) \prod_{i=2}^{n}P(x_i|x_{i-1}, x_{i-2}, ..., x_1) .$$

However, these conditional probabilities are hard to model for even medium-sized vocabularies. One simplification is to make a <em>first-order Markov assumption</em>, which says that the probability of a word given all previous words is equal to the probability of the word given just the one word before it, that is,

$$ P(x_i | x_1, x_2, ..., x_{i-1}) = P(x_i | x_{i-1}). $$

This allows us to factor

$$ P(x_1, x_2, ... x_n)  = \prod_{i=1}^{n}P(x_i|x_{i-1}) $$

where $x_0$ is a special "START" symbol.

The formulation above is called a <em>bigram</em> language model, since it is based on pairs of consecutive words. If the context is expanded to include the two previous words, we get a <em>trigram</em> model,
    
$$ P(x_1, x_2, ... x_n)  = \prod_{i=1}^{n}P(x_i|x_{i-2}, x_{i-1}) $$

where $x_{-1}, x_0$ are special start symbols.

In the same way, an <em>n-gram</em> model can be defined by expanding the context to include the $n-1$ previous words.

### Learning an $n$-gram model

<em>Maximum-likelihood estimation</em> can be used to learn an $n$-gram model. For this, we simply need to count the number of occurrences of each $n$-gram and $n-1$ gram in the corpus.

For instance, to train a trigram model, let $c(u,v,w)$ be the number of times that trigram $(u,v,w)$ is seen in the training corpus and let $c(u,v)$ be the number of times that bigram $(u,v)$ is seen. The maximum likelihood estimate of $P(w|u,v)$ is then $P(w|u,v) = \frac{c(u,v,w)}{c(u,v)}$. Of course, we might want to smooth this using Laplace smoothing or something similar.

The provided text file `shakespeare.txt` contains a selection of Shakepeare's sonnets and plays. It also contains stage directions, copyright notices, and various other things that we won't bother removing. We will read in a list of all sentences in the file.

Here is an example of one of the sentences.

`I grant thou wert not married to my muse, And therefore mayst without attaint o'erlook The dedicated words which writers use Of their fair subject, blessing every book.`

### <font color='blue'>Experimenting with text generation</font>

As an experiment, a bigram and a trigram model are fit to the provided text from Shakespeare. The text generated from each of the two models is shown below.

Bigram: `Go with flowing tides and mart , Catesby , and , you 'll have again , madam ; we will keep 't , Hews down His sword impress . JOHN . These good lord , the beaten ? Look 'd by their congeal again in state in this was done any he ? A little prating coxcomb ? Why should be more To withdraw yourself ; I come -O Jove would you stay , Like perspectives which nature reigned , For he might please . AGRIPPA , that ? MESSENGER . Here 's mistress , which I could control thee`

Trigram: `VALENTINE . Exeunt SCENE VIII . MESSENGER . TROILUS . Another way so happily ; the weight of a greater falseness ; Which were inshell 'd when Marcius stood for his offence . Then all alone beweep my outcast state , And deeper than e 'er my haps , my lord , The imminent death of the knee Where thrift may follow , Stephano ! O Imogen ! IMOGEN . DECIUS . We should be visited . Pardon , master ; red as Mars his idiot ! Do you hear the suit Of Count Orsino 's embassy . For God`

### Computing likelihoods

Log-likelihood of a sentence gives a concrete indication of why incorporating more context is helpful. For a given sentence $s$ with $n$ tokens $(x_1, \ldots, x_n)$, the log-likelihood is defined as
$$L(s) = \sum_{i=1}^{n}\log p(x_i|\mbox{context})$$
For a unigram model, this maximum-likelihood estimate of $p(x_i)$ would simply be
$$p(x_i) = \frac{c(x_i)}{N}$$
where N is the total number of words in the dataset and $c(x_i)$ is the number of occurrences of $x_i$. Similarly, for a bigram model, we have
$$p(x_i|\mbox{context}) = \frac{c(x_{i-1}, x_i)}{c(x_{i-1})}$$

As an example, the log-likelihood of the sentence `And on a love-book pray for my success?` under a unigram model and a bigram model is tabulated below.

| Model | Log-likelihood |
| --- | --- |
| Unigram | -65.86 |
| Bigram | -45.88 |

### Text Completion
Following is a test sample from the first sonnet 

`From fairest creatures we desire increase,
  That thereby beauty's rose might never die,
  But as the riper should by time decease,
  His tender heir might bear his memory:
  But thou contracted to thine own bright eyes,
  Feed'st thy light's flame with self-substantial fuel,
  Making a famine where abundance lies,
  Thy self thy foe, to thy sweet self too cruel:
  Thou that art now the world's fresh ornament,
  And only herald to the gaudy spring,
  Within thine own bud buriest thy content,
  And tender churl mak'st waste in niggarding:
    Pity the world, or else this glutton be,
    To eat the world's due, by the grave and thee.`

We take a prompt from the sample above `From fairest creatures we desire increase` and generate 20 words of text from the bigram and trigram model.

| Model | Generated text |
| --- | --- |
| Bigram | . Another part . Yet in the commons KING HENRY . NORTHUMBERLAND . Exeunt . ' Was beastly dumb and |
| Trigram | Naiads, testament- hypocrite! unseeing reward, cheeks! (sings) perdu!- that's-when? Samp. tallest! missing. native, basin uses. weakness. perform, incurable. braggarts Mall's |

There are various tweaks that would improve this model: for instance, careful <em>smoothing</em> and using longer-range dependencies (via variable-length Markov models such as <em>probabilistic suffix trees</em>). The next big boost in performance would come from replacing tabular estimates of conditional probabilities $P(x_i|x_1, ...., x_{i-1})$ by <em>recurrent neural nets</em>.
