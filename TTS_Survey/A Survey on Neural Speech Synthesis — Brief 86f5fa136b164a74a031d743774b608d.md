# A Survey on Neural Speech Synthesis — Brief

> *Text to Speech (TTS), also known as speech synthesis, aims to synthesize intelligible and natural speech from text.*
> 

## 🧊Reference

[A%20Survey%20on%20Neural%20Speech%20Synthesis%20%E2%80%94%20Brief%2086f5fa136b164a74a031d743774b608d/TTS%25E7%25BB%25BC%25E8%25BF%25B0.pdf](A%20Survey%20on%20Neural%20Speech%20Synthesis%20%E2%80%94%20Brief%2086f5fa136b164a74a031d743774b608d/TTS%25E7%25BB%25BC%25E8%25BF%25B0.pdf)

[TTS综述](A%20Survey%20on%20Neural%20Speech%20Synthesis%20%E2%80%94%20Brief%2086f5fa136b164a74a031d743774b608d/TTS%25E7%25BB%25BC%25E8%25BF%25B0.pdf)

[https://tts-tutorial.github.io/interspeech2022/INTERSPEECH_Tutorial_TTS.pdf](https://tts-tutorial.github.io/interspeech2022/INTERSPEECH_Tutorial_TTS.pdf)

[INTERSPEECH_Tutorial_TTS.pdf](https://tts-tutorial.github.io/interspeech2022/INTERSPEECH_Tutorial_TTS.pdf)

## 🚀Introduction

### History of TTS Technology

- **Articulatory Synthesis**
    - produces speech by simulating the behavior of human articulators such as lips, tongue, glottis, and moving vocal tract.
    
    🙂pros
    
    - Ideally most effective speech synthesis
    
    😅cons
    
    - Very difficult to model articulator behavior in practice: hard to collect the data for articulator simulation
- **Formant Synthesis**
    - produces speech based on a set of rules that
    control a simplified source-filter model.
    - synthesized by an additive synthesis module and an acoustic model with varying parameters like fundamental frequency, voicing, and noise levels.
    
    🙂pros
    
    - can produce highly intelligible speech
    
    😅cons
    
    - does not rely on a large human corpus as in the concatenative synthesis
    - sounds less natural and has artifacts
    - difficult to specify rules for synthesis
- **Concatenative Synthesis**
    - relies on the concatenation of pieces of speech that are stored in a database
        - the database consists of speech units ranging from whole sentence to syllables that are recorded by voice actors
    - Searches speech units to match the given input text, and produces speech waveform by concatenating these units together
    
    🙂pros
    
    - can generate audio with high intelligibility and authentic timbre close to the original voice actor
    
    😅cons
    
    - requires a huge database to cover all possible combinations of speech units for spoken words
    - generated voice is less natural and emotional since concatenation can result in less smoothness in stress, emotion, prosody, etc.
- **Statistical Parametric Synthesis (SPSS)**
    - First, generate the acoustic parameters that are necessary to produce speech
    - then recover speech from the generated acoustic parameters using some algorithms
    - consists of three components:
        - A text analysis module
            - process the text: text normalization, grapheme to phoneme conversion, word segmentation, etc.
            - then extract phoneme, duration, and POS tags from which different granularities.
        - A parameter prediction module (acoustic model with hidden Markov base)
            - trained with the paired linguistic features and parameters (acoustic features): including fundamental frequency, spectrum or cepstrum, etc.
        - A vocoder analysis/synthesis module (vocoder)
            - synthesize speech from the predicted acoustic features
    
    🙂pros
    
    1. the audio is more natural
    2. convenient to modify the parameters to control the generated speech
    3. requires fewer recordings than concatenative synthesis
    
    😅cons
    
    1.  lower intelligibility due to artifacts such as muffled, buzzing, or noisy audio.
    2. generated voice is still robotic and can be easily differentiated from human recording speech.
- 🕋**Neural Speech Synthesis**
    - adopts neural networks as the model backbone for speech synthesis.
    - early models are adopted in SPSS to replace HMM for acoustic modeling
    
    <aside>
    💡 WaveNet proposed: directly generate waveform from linguistic features
    
    </aside>
    
    - Deepvoice1/2 still follows SPSS
    - End-to-end models are proposed: **Tacotron1/2**, **Deepvoice3**, **FastSpeech1/2**
    - Fully end-to-end TTS systems are developed to directly generate waveform from the text: **ClariNet**, **FastSpeech2s** and **EATS**
    
    🙂pros
    
    1. high voice quality in terms of both intelligibility and naturalness
    2. less requirement on human preprocessing and feature development

### Organization of This Survey

- **Key components in TTS**
    
    ![Untitled](A%20Survey%20on%20Neural%20Speech%20Synthesis%20%E2%80%94%20Brief%2086f5fa136b164a74a031d743774b608d/Untitled.png)
    
    Text analysis: converts a text sequence into linguistic features
    
    Acoustic models: generate acoustic features from linguistic features
    
    Vocoders: Synthesize waveform from acoustic models
    
    Fully end-to-end models: directly convert characters/phonemes into waveforms
    
    ![Untitled](A%20Survey%20on%20Neural%20Speech%20Synthesis%20%E2%80%94%20Brief%2086f5fa136b164a74a031d743774b608d/Untitled%201.png)
    
- **Advanced topics in TTS**
    - Speed up the autoregressive generation and reduce the model size
    - Build data-efficient TTS models under low resource settings
    - Improve the Robustness of speech synthesis
    - Model, control, and transfer the style/prosody of speech in order to generate expressive speech
    - Efficient voice adaptation with limited adaptation data and parameters is critical for practical TTS applications

## 🔥Key Components in TTS

### Main Taxonomy

![Untitled](A%20Survey%20on%20Neural%20Speech%20Synthesis%20%E2%80%94%20Brief%2086f5fa136b164a74a031d743774b608d/Untitled%202.png)

**Data flow**

1. **Characters**: raw format of text
2. **Linguistic features**:  obtained through text
analysis and contain rich context information about pronunciation and prosody
    
    <aside>
    💡 **Phonemes** are one of the most important elements in linguistic features and are usually used alone to represent text in neural-based TTS models
    
    </aside>
    
3. **Acoustic features**: abstractive representations of speech waveforms
    - In SPSS, LSP (line spectral pairs), MCC (Mel-cepstral coefficients), MGC (Mel-generalized coefficients), F0, and BAP (band aperiodicities) are used as acoustic features, which can be easily converted into waveform through vocoders such as STRAIGHT and WORLD
    
    <aside>
    💡 In neural-based end-to-end TTS models, M**el-spectrograms or linear-spectrograms** are usually used as acoustic features, which are converted into waveforms using **neural-based vocoders**.
    
    </aside>
    
4. **Waveform**: the final format of speech.

### Other Taxonomies

1) **Autoregressive or non-autoregressive**.

2) **Generative model** (normal sequence generation, flow, GAN, VAE, diffusion models)

3) **Network structure** (CNN, RNN, self-attention, and hybrid structures)

![Untitled](A%20Survey%20on%20Neural%20Speech%20Synthesis%20%E2%80%94%20Brief%2086f5fa136b164a74a031d743774b608d/Untitled%203.png)

### 🗒️Text Analysis

> ***Text analysis**, also called frontend in TTS, transforms input text into linguistic features that contain rich information about **pronunciation and prosody** to ease speech synthesis.*
> 
- **Text normalization**: converts raw written text (non-standard words) into spoken form words through text normalization
    - early works are rule-based
    - neural networks are leveraged to model text normalization as a sequence-to-sequence task: source and target sequences are non-standard words and spoken-form words respectively.
    - recent works propose to combine both rule-based and neural-based models
- **Word Segmentation**: detect the word boundary for raw text
    - ensure the accuracy for later POS tagging, prosody prediction, and grapheme to the phoneme conversion process
- **Part of Speech (POS) tagging**: tag part-of-speech of each word such as noun, verb, or preposition.
- **Prosody prediction**: relies on tagging systems to label each kind of prosody
    
    <aside>
    💡 The prosody information, such as rhythm, stress, and intonation of speech,
    corresponds to the variations in syllable duration, loudness and pitch
    
    </aside>
    
    - English: ToBI (tones and break indices)
    - Chinese:  PW (prosodic word), PPH (prosodic phrase) and IPH (intonational phrase) construct three-layer prosody tree
        - Some works investigate different model structures such as CRF, RNN, and self-attention for prosody prediction in Chinese.
- **Grapheme-to-phoneme (G2P) conversion**: converts character (grapheme) into pronunciation (phoneme) can greatly ease speech synthesis
    - A manually collected **grapheme-to-phoneme lexicon** is usually leveraged for conversion.
        - For alphabetic languages, lexicon cannot cover the pronunciations of all the words.  G2P conversions for English is mainly responsible to **generate the pronunciations of all the words**
        - For languages like Chinese, there are a lot of **polyphones** that can be only decided according to the context of a character. G2P conversions in this kind of language is mainly responsible for **polyphone disambiguation**

<aside>
🗣 Although text analysis seems to receive less attention in neural TTS compared to SPSS, it has been incorporated into neural TTS in various ways:

1) **Multi-task and unified frontend model**: to cover all the tasks in text analysis in a multi-task paradigm and achieve good results. 

2) **Prosody prediction**. 

Prosody is critical for the naturalness of speech synthesis. 

Although neural TTS models simplify the text analysis module, some features for prosody prediction are incorporated into text encoders, such as the prediction of pitch, duration, phrase break, breath, or filled pauses are built on top of the text (character or phoneme) encoder in TTS models.

Some other ways to incorporate prosody features include 

1) reference encoders that learn the prosody representations from reference speech

2) text pre-training that learns good text representations with implicit prosody information through self-supervised pre-training 

3) incorporating syntax information through dedicated modeling methods such as graph networks 

</aside>

### ⛩️Acoustic models

![Untitled](A%20Survey%20on%20Neural%20Speech%20Synthesis%20%E2%80%94%20Brief%2086f5fa136b164a74a031d743774b608d/Untitled%204.png)

- **Brief**
    
    > ***Acoustic models** generate acoustic features from linguistic features or directly from phonemes or characters*
    > 
    - Early HMM and DNN-based models in SPSS
    - Sequence to sequence models based on an encoder-attention-decoder framework (LSTM, CNN, and self-attention)
    - Latest feed-forward networks (CNN or self-attention) for parallel generation
    
    ---
    
    **Different kinds of acoustic features have been tried** 
    
    - Mel-cepstral coefficients (MCC)
    - Mel-generalized coefficients (MGC)
    - Band aperiodicity (BAP)
    - Fundamental frequency (F0)
    - Voiced/Unvoiced (V/UV)
    - Bark-frequency cepstral coefficients (BFCC)
    
    <aside>
    🎉 And the most widely used: **Mel-spectrogram**
    
    </aside>
    
    ---
    
- **Acoustic models in SPSS**
    
    Developments are driven by several considerations:
    
    1. taking more context information as input
    2. modeling the correlation between output frames
    3. better combating the over-smoothing prediction problem, since the mapping from linguistic features to acoustic features is one to many
    - **HMM**
        - Generate speech parameters where the observation vectors of HMM consist of spectral parameter vectors such as MCC and F0.
        - Is more flexible in changing speaker identities, emotions, and speaking styles
        
        <aside>
        🧐 One major drawback: the quality of synthesized speech is not good enough!
        
        </aside>
        
        Two reasons:
        
        1. The accuracy of acoustic models is not good, and the predicted acoustic features are over-smoothing and lack details.
            1. due to lack of modeling capacity in HMM (thus propose DNN/LSTM/CBHG/VoiceLoop/GAN/E2E-Attention based models)
        2. The vocoding techniques are not good enough 
    
    ---
    
- 🔥**Acoustic models in End-to-End TTS**
    
    Advantages:
    
    1. Sequence to sequence models implicitly learn the alignments through attention or predict the duration jointly, which are more end-to-end and require less processing
    2. Linguistic features are simplified into only character or phoneme sequence, and the acoustic features have changed from low-dimensional and condensed cepstrum to high-dimensional Mel-spectrograms or even more high-dimensional linear-spectrograms
    - 🎆**RNN-based models (Tacotron Series)**
        
        
        | Tacotron | leverages an encoder-attention-decoder framework and takes characters as input and outputs linear spectrograms, and uses the Griffin-Lim algorithm to generate waveforms. |
        | --- | --- |
        | Tacotron 2 | generates Mel-spectrograms and converts Mel-spectrograms into waveforms using an additional WaveNet model |
        | GST-Tacotron                  Ref-Tacotron    | uses a reference encoder and style tokens to enhance the expressiveness of speech synthesis |
        | DurIAN                                        Non-attentative Tacotron     | Removing the attention mechanism in Tacotron, and instead using a duration predictor for autoregressive prediction |
        | Parallel Tacotron 1/2 | Changing the autoregressive generation in Tacotron to a non-autoregressive generation |
        | Wave-Tacotron | Building end-to-end text-to-waveform models based on Tacotron |
    - 🌏**CNN-based models (DeepVoice Series)**
        
        
        | DeepVoice | obtains linguistic features through neural networks and leverages a WaveNet vocoder to generate waveforms |
        | --- | --- |
        | DeepVoice2  | follows DeepVoice with improved network structures and multi-speaker modeling. First generates linear-Spectrograms using Tacotron and generates waveform using WaveNet. |
        | DeepVoice3 | Leverages a fully-connected network structure for speech synthesis and generates Mel-spectrograms from characters. Uses sequence-to-sequence model and directly predicts Mel-spectrograms. |
        | ClariNet | Generate waveforms from texts in a fully end-to-end way |
        | ParaNet | Fully convolutional networks based non-autoregressive model can speed up the Mel-spectrogram generation and obtain reasonably good speech quality |
        | DCTTS | leverages a fully convolutional-based encoder-attention-decoder network to generate Mel-spectrograms from character sequences. Then uses a spectrogram super-resolution network to obtain liner-spectrograms, and synthesize waveform using Griffin-Lim |
    - 🚇**Transformer-based models (FastSpeech Series)**
        - Leverages Transformer-based encoder-attention-decoder architecture to generate Mel-spectrograms from phonemes.
        - RNN-based encoders encoder-attention-decoder models like Tacotron2 suffer from the following two issues:
            - RNN-based encoder and decoder **cannot be trained in parallel**, and RNN-based encoder **cannot be put in parallel in inference**, which affects the efficiency both in training and inference
            - RNN is not good at **modeling the long dependency** in long text and speech sequences
        
        ---
        
        **TransformerTTS:**
        
        - Adopts the basic model structure of Transfomer and absorbs some designs from Tacotron 2 such as decoder pre-net/post-net and stop token prediction
        
        😍pros
        
        - Similar voice quality with Tacotron 2 and faster training time
        
        😅cons
        
        - the encoder-decoder attention in Transformer is not robust due to parallel computation
        
        ---
        
        **MutilSpeech:**
        
        - improves the robustness of the attention mechanism through encoder normalization, decoder bottleneck, and diagonal attention constrain
        
        ---
        
        **RobuTrans:** 
        
        - leverages duration prediction to enhance the robustness in autoregressive generation
        
        ---
        
        Previous neural-based models such as Tacotron 2, DeepVoice 3, and TransformerTTS all adopt autoregressive generation, which suffers from several issues:
        
        1. Slow inference speed
        2. Robust issues
        
        <aside>
        🔥 **FastSpeech** is proposed to solve these issues
        
        </aside>
        
        **FastSpeech:**
        
        - leverages an explicit duration predictor to expand the phoneme hidden sequence to match the length of Mel-spectrograms
            - How to get the duration label to train the duration predictor is critical for the prosody and quality of generated voice
        
        ---
        
        **FastSpeech2:**
        
        - Using ground-truth Mel-spectrograms as training targets. instead of distilled Mel-spectrograms from an autoregressive teacher model
            - **Simplifies** the two-stage teacher-student distillation pipeline in FastSpeech and also **avoids the information loss** in target Mel-spectrograms after distillation
            - **Providing more variance information** such as pitch, duration, and energy as decoder input, which **eases** the one-to-many mapping problem in text-to-speech
            - Achieves **better voice quality** and maintains the advantages of **fast, robust, and controllable** speech synthesis in FastSpeech
        
        ---
        
    - 🎂**Other Acoustic Models (Flow, GAN, VAE, Diffusion)**
        - **Flow-based models**
            
            
            | Flowtron | an autoregressive flow-based Mel-spectrogram generation model |
            | --- | --- |
            | Flow-TTS        Glow-TTS | leverage generative flow for non-autoregressive Mel-spectrogram generation |
        - **VAE-based models**
            
            **GMVAE-Tacotron**
            
            **VAE-TTS**
            
            **BVAE-TTS**
            
        - **GAN-based models**
            
            **GAN exposure**
            
            **TTS-Stylization**
            
            **Multi-SpectroGAN** 
            
        - **Diffusion-based models**
            
            **Diff-TTS**
            
            **Grad-TTS**
            
            **PriorGrad**
            
    
    ---
    

### 🎧Vocoders

<aside>
💡 The development of vocoders can be categorized into two stages:

1. the vocoders used in statistical parametric speech synthesis
2. **the neural network-based vocoders**
</aside>

![Untitled](A%20Survey%20on%20Neural%20Speech%20Synthesis%20%E2%80%94%20Brief%2086f5fa136b164a74a031d743774b608d/Untitled%205.png)

- **Some popular vocoders in SPSS include STRAIGHT and WORLD**
    - which consists of **vocoder analysis** and **vocoder synthesis** steps.
        - In vocoder analysis, it analyzes speech and gets acoustic features such as Mel-cepstral coefficients, band aperiodicity, and F0. In vocoder synthesis, it generates speech waveform from these acoustic features.
    
    ---
    
- 📢**Neural network-based vocoders**
    - 🍇**Autoregressive Vocoders:**
        
        **WaveNet**:
        
        - the first neural-based vocoder, which leverages **dilated convolution** to generate waveform points autoregressively. **Purely relies on end-to-end learning.**
        - originally generates speech waveform conditioned on **linguistic features**, and can be easily adapted to condition on **linear-spectrograms** and **Mel-spectrograms**
        
        😅cons:
        
        **Suffers from slow inference speed**
        
        ---
        
        **SampleRNN/Char2Wav**
        
        - SampleRNN leverages a **hierarchical recurrent network** for **unconditional** waveform generation
        - further integrated into Char2Wav to generate waveform **conditioned** on acoustic features
        
        ---
        
        **WaveRNN**
        
        - using a **recurrent neural network** and leveraging several designs including **dual softmax layer**, **weight pruning,** and **sub-scaling** techniques to reduce the computation
        
        ---
        
        **LPC-Net**
        
        - introduces **conventional digital signal processing** into neural networks
        - ⭐uses **linear prediction coefficients** to calculate the next waveform point while leveraging a **lightweight RNN** to compute the residual.
        - ⭐generates speech waveform conditioned on **BFCC features**, and can be easily adapted on **Mel-spectrograms**.
        - following works further improve LPCNet from different perspectives, such as **reducing complexity** for speedup and improving **stability for better quality**
        
        ---
        
    - 🕶️**Flow-based Vocoders:**
        
        > *Normalizing flow is a kind of generative model that transforms a **probability density** with a **sequence of invertible mappings**.*
        > 
        
        ![截屏2022-09-18 15.35.48.png](A%20Survey%20on%20Neural%20Speech%20Synthesis%20%E2%80%94%20Brief%2086f5fa136b164a74a031d743774b608d/%25E6%2588%25AA%25E5%25B1%258F2022-09-18_15.35.48.png)
        
        we can get a standard/normalized probability distribution through the sequence of invertible mappings based on the change-of-variables rules
        
        - During sampling, it generates data from a **standard probability distribution** through the inverse of these transforms
        - The flow-based models used in neural TTS can be divided into two categories according to the two different techniques:
        
        ---
        
        - 🏂**Autoregressive transforms:**
            
            **Inverse autoregressive flow (IAF)**
            
            Can be regarded as a **dual formation** of autoregressive flow (AF). The sampling in IAF is parallel while the inference for likelihood estimation is sequential.
            
            - **Parallel WaveNet:**
                
                Leverages **probability density distillation** to marry the efficient sampling of IAF with the efficient training of AR modeling
                
            - **ClariNet:**
                
                Uses IAF and **teacher distillation**, and leverages a **closed-form KL divergence** to simplify and stabilize the distillation process.
                
            
            <aside>
            💡 Although Parallel Wavenet and ClariNet can generate speech in parallel, it relies on **sophisticated teacher-student training** and still requires **large computation**.
            
            </aside>
            
        
        ---
        
        - 🐠**Bipartite transforms:**
            
            **Glow or RealNVP**
            
            bipartite transforms leverage the **affine coupling** that ensures the **output can be computed from the input** and vice versa. Some vocoders based on bipartite transforms include:
            
            - **WaveGlow**
            - **FloWaveNet**
            
            Which achieves high voice quality and fast inference speed
            
        
        ---
        
        <aside>
        💡 Both autoregressive and bipartite transforms have their **advantages** and **disadvantages**:
        
        1. Autoregressive transforms are **more expressive** than bipartite transforms by modeling dependency between **data distribution x** and **standard probability distribution z**, but require **teacher distillation** that is complicated in training
        2. Bipartite transforms enjoy **a much simpler training pipeline**, but usually, require a larger number of parameters to reach comparable capacities with autoregressive models 
        </aside>
        
        ---
        
        - To combine the advantages of both autoregressive and bipartite transforms:
        
        **WaveFlow:** 
        
        Provides a unified view of **likelihood-based models** for audio data to explicitly trade i**nference parallelism** for **model capacity**
        
        <aside>
        💡 In this way, **WaveNet**, **WaveGlow**, and **FloWaveNet** can be regarded as special cases of **WaveFlow**
        
        </aside>
        
    - 🎓**GAN-based Vocoders:**
        
        > ***GAN** consists of a **generator** for data generation, and a **discriminator** to judge the authenticity of the generated data.*
        > 
        
        ![Untitled](A%20Survey%20on%20Neural%20Speech%20Synthesis%20%E2%80%94%20Brief%2086f5fa136b164a74a031d743774b608d/Untitled%206.png)
        
        ---
        
        - 🌮**Generator**
            - Most GAN-based vocoders use **dilated convolution** to increase the **receptive field** to model the **long dependency** in the waveform sequence
            - use **transposed convolution** to upsample the condition information (**linguistic features** to **Mel-spectrograms**) to match the length of the waveform sequence
            - Some vocoders choose to **iteratively upscale the condition information**  and perform **dilated convolution**, which can avoid too long sequences in lower layers
            
            **VocGAN**
            
            Proposes a **multi-scale generator** that can gradually output waveform sequences at different scales, from **coarse-grained** to **fine-grained**.
            
            **Hifi-GAN**
            
            processes different patterns of various lengths in parallel through a **multi-receptive field fusion module**, and also has the flexibility to trade off between **synthesis efficiency** and **sample quality**
            
        
        ---
        
        - 🍔**Discriminator**
            - 🗼**Random window discriminators**
                
                **GAN-TTS:**
                
                use **multiple discriminators**, where each is feeding with **different random windows of waveform** with and without conditional information. 
                
                <aside>
                💡 **Random window discriminators** have several benefits:
                
                - evaluates audios in a different **complementary way**
                - simplifies the **true/false judgments** compared with full audio
                - acts as a **data augmentation** effect
                </aside>
                
            - ⛲**Multi-scale discriminators**
                
                **MelGAN:**
                
                use **multiple discriminators** to judge **audios on different scales** (different downsampling ratios compared with original audio). 
                
                <aside>
                💡 The advantage of **multi-scale discriminators** is that the discriminator in each scale can **focus on the characteristics** in different frequency ranges.
                
                </aside>
                
            - 🎏**Multi-period discriminators**
                
                **Hifi-GAN:**
                
                leverages **multiple discriminators**, where each accepts **equally spaced samples** of input audio with a period. 
                
                Specifically, the 1D waveform sequence with a length of *T* is reshaped into 2D data [*p, T /p*] where *p* is the period, and processed by a 2D convolution. 
                
                <aside>
                💡 **Multi-period discriminators** can **capture different implicit structures** by looking at different parts of input audio in different periods.
                
                </aside>
                
            - 🏖️**Hierarchical discriminators**
                
                **VocGAN**
                
                judges the generated waveform in **different resolutions** from **coarse-grained** to **fine-grained**
                
                <aside>
                💡 **Hierarchical discriminators** can guide the generator to learn the mapping between the acoustic features and waveform in both **low and high frequencies.**
                
                </aside>
                
        
        ---
        
        - 🍲**Loss**
            
            
            **Regular GAN losses**
            
            - WGAN-GP
            - hinge-loss GAN
            - LS-GAN
            
            **Other specific losses**
            
            - STFT loss
            - feature matching loss
            
            <aside>
            💡 The additional losses can improve the **stability** and **efficiency** of adversarial
            training and improve the **perceptual audio quality**.
            
            </aside>
            
        
        ---
        
    - 🤯**Diffusion-based Vocoders:**
        
        **DiffWave** **WaveGrad** **PriorGrad:**
        
        leverage denoising diffusion probabilistic models (**DDPM** or **Diffusion**)
        
        - The basic idea is to formulate the **mapping between data and latent distributions** with the **diffusion process** and **reverse process**:
            - in the **diffusion process**, the waveform data sample is gradually added with some **random noises** and finally becomes **Gaussian noise**;
            - in the **reverse process**, the random Gaussian noise is gradually **denoised** into waveform data sample **step by step**.
        
        <aside>
        💡 Diffusion-based vocoders can generate speech with **very high voice quality** but suffer from **slow inference speed** due to the **long iterative process**.
        
        </aside>
        
        ---
        
    - Other vocoders
        
        Some works leverage a **neural-based source-filter model** for waveform generation, aiming to achieve **high voice quality** while maintaining **controllable speech generation**. 
        
        ---
        
    
    <aside>
    🗣️ Characteristics
    
    - In terms of **mathematical simplicity**, autoregressive (**AR**) based
    models are easier than other generative models such as **VAE**, **Flow**, **Diffusion**, and **GAN**.
    - All the generative models except **AR** can **support parallel speech generation**.
    - Except for **AR** models, all generative models can **support latent manipulations** to some extent
        - (some **GAN-based** vocoders do not take random Gaussian noise as model input, and thus cannot support latent manipulation)
    - **GAN-based** models **cannot estimate the likelihood of data samples**, while other models enjoy this benefit.
    </aside>
    
    ![Untitled](A%20Survey%20on%20Neural%20Speech%20Synthesis%20%E2%80%94%20Brief%2086f5fa136b164a74a031d743774b608d/Untitled%207.png)
    

### 🏆Towards Fully End-to-end TTS

![Untitled](A%20Survey%20on%20Neural%20Speech%20Synthesis%20%E2%80%94%20Brief%2086f5fa136b164a74a031d743774b608d/Untitled%208.png)

<aside>
💡 Fully end-to-end TTS models can generate speech waveforms from character or phoneme sequences directly, which has the following advantages:

1. It requires **less human annotation** and **feature development** (alignment between text and speech)
2. The joint end-to-end optimization can avoid **error propagation** in cascaded models
3. It can also reduce the **training, development, and deployment cost**
</aside>

- There are big challenges to training TTS models in an end-to-end way, mainly due to the **modalities** between **text** and **speech waveform**
- The development of Neural TTS follows a progressive process toward fully e2e models:
    - Simplifying **text analysis** module and **linguistic features**
        
        In end-to-end models, only the **text normalization** and **grapheme-to-phoneme** conversion are **retained** to convert characters into phonemes, or the whole text analysis module is removed by directly taking **characters as input**
        
    - Simplifying **acoustic features**, where the complicated acoustic features such as MGC, BAP, and F0 used in SPSS are **simplified into Mel-spectrograms**.
    - Replacing two or three modules with a **single end-to-end model**
        
        For example, the acoustic models and vocoders can be replaced with a single vocoder model such as WaveNet.
        

![Untitled](A%20Survey%20on%20Neural%20Speech%20Synthesis%20%E2%80%94%20Brief%2086f5fa136b164a74a031d743774b608d/Untitled%209.png)

- **🙋‍♂️stage 0:**
    
    Statistic parametric synthesis uses three basic modules, where the **text analysis module** converts characters into linguistic features, **acoustic models** generate acoustic features from linguistic features (where the target acoustic features are obtained through vocoder analysis), and then **vocoders** synthesize speech waveforms from acoustic features through parametric calculation.
    
- 👏**stage 1:**
    
    combine the **text analysis** and **acoustic model** in SPSS into an **end-to-end acoustic model** that directly generates acoustic features from phoneme sequence, and then uses a **vocoder** in SPSS to generate the waveform.
    
- 🐘**stage 2:**
    
    **WaveNet** 
    
    is first proposed to directly generate **speech waveform** from **linguistic features,** which can be regarded as a combination of an **acoustic model** and a **vocoder**. This kind of models still require a **text analysis module** to generate linguistic features.
    
- 🎁**stage 3:**
    
    **Tacotron**
    
    is further proposed to **simplify linguistic** and **acoustic features**, which directly predicts **linear spectrograms** from **characters/phonemes** with an **encoder-attention-decoder** model, and converts linear spectrograms into **waveform** with **Griffin-Lim**. 
    
    ---
    
    The following works such as 
    
    **DeepVoice 3 Tacotron 2 TransformerTTS FastSpeech 1/2** 
    
    predict **Mel-spectrograms** from **characters/phonemes** and further use a **neural vocoder** such as 
    
    **WaveNet WaveRNN WaveGlow FloWaveNet FloWaveNet**
    
    to generate waveforms.
    
- 🎳**stage 4:**
    
    Some fully end-to-end TTS models are developed for direct text-to-waveform synthesis. 
    
    ---
    
    **Char2Wav** 
    
    leverages an **RNN-based encoder-attention-decoder** model to generate acoustic features from characters and then uses **SampleRNN** to generate waveforms. The two models are **jointly tuned** for direct speech synthesis. 
    
    **ClariNet**
    
    jointly tunes an **autoregressive acoustic model** and a **non-autoregressive vocoder** for direct waveform generation.
    
    **FastSpeech 2s** 
    
    directly generate speech from text with a **fully parallel structure**, which can greatly speed up inference. To alleviate the difficulty of **joint text-to-waveform training**, it leverages an auxiliary **Mel-spectrogram decoder** to help learn the **contextual representations** of phoneme sequences. 
    
    **EATS**
    
    also directly generates waveforms from characters/phonemes, which leverages **duration interpolation** and **soft dynamic time wrapping loss** for end-to-end **alignment learning**. 
    
    **Wave-Tacotron**
    
    builds a **flow-based decoder** on **Tacotron** to directly generate waveforms, which uses **parallel waveform generation** in the **flow part** but still an **autoregressive generation** in the **Tacotron part**.
    

## 🤿Advanced Topics in TTS (🥲Hardcore Tricks)

### 🫑**Background and Taxonomy**

![Untitled](A%20Survey%20on%20Neural%20Speech%20Synthesis%20%E2%80%94%20Brief%2086f5fa136b164a74a031d743774b608d/Untitled%2010.png)

![Untitled](A%20Survey%20on%20Neural%20Speech%20Synthesis%20%E2%80%94%20Brief%2086f5fa136b164a74a031d743774b608d/Untitled%2011.png)

### 🌺**Fast TTS**

- **Preview**
    
    <aside>
    💡 Text-to-speech synthesis systems are usually deployed in cloud servers or embedded devices, which require fast synthesis speed. However, early neural TTS models usually adopt **autoregressive Mel-spectrogram** and **waveform generation**, which are **very slow** considering the long speech sequence.
    
    </aside>
    
    To solve this problem, different techniques have been leveraged to speed up the inference of TTS models, 
    
    1) **non-autoregressive generation** that generates Mel-spectrograms and waveform in parallel; 
    
    2) **lightweight** and **efficient** model structure;
    
    3) techniques leveraging the **domain knowledge of speech** for fast speech synthesis. 
    
    We introduce these techniques as follows.
    
- 🍁**Parallel Generation**
    
    ![Untitled](A%20Survey%20on%20Neural%20Speech%20Synthesis%20%E2%80%94%20Brief%2086f5fa136b164a74a031d743774b608d/Untitled%2012.png)
    
    <aside>
    💡 Table 8 summarizes typical modeling paradigms, the corresponding TTS models, and time complexity in training and inference. As can be seen, TTS models that use **RNN-based autoregressive models are slow** in both training and inference, with *O*(*N*) computation, where *N* is the sequence length.
    
    </aside>
    
    To avoid the slow training time caused by the RNN structure, 
    
    - **DeepVoice 3** and **TransformerTTS** leverage **CNN** or a **self-attention-based structure** that can support **parallel training** but still require **autoregressive inference.**
    
    ---
    
    To speed up inference, 
    
    - **FastSpeech 1/2** designed a **feed-forward Transformer** that leverages a **self-attention structure** for both **parallel training** and **inference**, where the computation is reduced to **O(1)**.
    
    ---
    
    - Most **GAN-based models** for Mel-spectrogram and waveform generation are **non-autoregressive**, with **O(1)** computation in both **training** and **inference**.
    
    ---
    
    - **Parallel WaveNet** and ClariNet leverage **inverse autoregressive flow** [169], which enable **parallel inference** but require **teacher distillation** for **parallel training**.
    - **WaveGlow** and **FloWaveNet** leverage **generative flow** for **parallel training** and **inference**. However, they usually need to **stack multiple flow iterations** T to ensure the quality of the mapping between **data** and **prior distributions**.
    
    ---
    
    - Similar to **flow-based models**, **diffusion-based models** also require **multiple diffusion steps** T in the forward and reverse process, which **increases the computation**.
- 🥪**Lightweight Model**
    
    <aside>
    💡 While **non-autoregressive generation** can fully leverage the parallel computation for **inference speedup**, the number of **model parameters** and **total computation cost** are not reduced, which makes it slow when deploying on **mobile phones** or e**mbedded devices** since the parallel computation capabilities in these devices are not powerful enough. Therefore, we need to design **lightweight** and **efficient models** with less computation cost for inference speedup, even using autoregressive generation.
    
    </aside>
    
    - Some widely used techniques for designing lightweight models include **pruning**, **quantization**, **knowledge distillation**, **neural architecture search**, etc.
    - **WaveRNN** uses techniques like **dual softmax**, **weight pruning**, and **subscale prediction** to speed up inference.
    - **LightSpeech** leverages **neural architecture search** to find lightweight
    architectures to further speed up the inference of **FastSpeech 2** [292] by 6.5x while maintaining voice quality.
    - **SqueezeWave** leverages **waveform reshaping** to reduce the **temporal length** and replaces the 1D convolution with **depthwise separable convolution** to reduce computation cost while achieving similar audio quality.
    - **Kanagawa and Ijima** compress the model parameters of **LPCNet** with
    **tensor decomposition**.
    - **Hsu and Lee** propose a heavily compressed **flow-based model** to reduce
    **computational resources**, and a **WaveNet-based** **post-filter** to maintain audio quality.
    - **DeviceTTS** leverages the model structure of **DFSMN** and **mix-resolution decoder** to predict multiple frames in one decoding step to speed up inference.
    - **LVCNet** adopts a **location-variable convolution** for different waveform intervals, where the convolution coefficients are predicted from Mel-spectrograms. It speeds up the **Parallel WaveGAN** vocoder by 4x without any degradation in sound quality.
    - Wang et al. propose a **semi-autoregressive mode** for Mel-spectrogram generation, where the Mel spectrograms are generated in an **autoregressive mode** for **individual phonemes** and a **non-autoregressive mode** for **different phonemes**.
- 🕵️**Speedup with Domain Knowledge**
    
    <aside>
    💡 **Domain knowledge** from speech can be leveraged to speed up inference, such as **linear prediction**, **multiband modeling**, **subscale prediction**, **multi-frame prediction**, **streaming synthesis**, etc.
    
    </aside>
    
    - **LPCNet** combines **digital signal processing** with **neural networks**, by using **linear prediction coefficients** to calculate the next waveform and a **lightweight model** to predict the **residual value**, which can speed the inference of autoregressive waveform generation.
    
    <aside>
    💡 Another technique that is widely used to speed up the inference of **vocoders** is **subband modeling**, which divides the waveform into **multiple subbands** for fast inference.
    
    </aside>
    
    - Typical models include **DurIAN**, **multi-band MelGAN**, **subband WaveNet**, and **multi-band LPCNet**.
    - **Bunched LPCNet** reduces the computation complexity of **LPCNet** with **sample** **bunching** and **bit bunching**, achieving more than 2x speedup.
    - **Streaming TTS** synthesizes speech once some input tokens are coming, without waiting for the whole input sentence, which can also speed up inference.
    - **FFTNet** uses a simple architecture to mimic the **Fast Fourier Transform (FFT),** which
    can generate audio samples in real-time. Okamoto et al. further enhance **FFTNet** with **noise shaping** and **subband techniques**, improving the voice quality while keeping a **small model size**.
    - **Popov et al.** propose **frame splitting** and **cross-fading** to synthesize some parts of the waveform **in parallel** and then concatenate the synthesized waveforms together to ensure fast synthesis on **low-end devices**.
    - **Kang et al.** accelerate **DCTTS** with **network reduction** and **fidelity improvement** techniques such as **group highway activation**, which can synthesize speech in real time with a single CPU thread.

### **Low-Resource TTS**

![Untitled](A%20Survey%20on%20Neural%20Speech%20Synthesis%20%E2%80%94%20Brief%2086f5fa136b164a74a031d743774b608d/Untitled%2013.png)

<aside>
💡 Building high-quality TTS systems usually requires a large amount of high-quality **paired text and speech data**. However, there are more than 7,000 languages in the world, and most languages lack training data for developing TTS systems. As a result, popular commercialized speech
services can only support dozens of languages for TTS. Supporting TTS for low-resource languages can not only have business value but is also beneficial for social good. Thus, a lot of research works build TTS systems under low data resource scenarios. We summarize some representative techniques for low-resource TTS in Table 9 and introduce these techniques as follows.

</aside>

- **Self-supervised training**.
    
    Although paired text and speech data are hard to collect, unpaired speech and text data (especially text data) are relatively easy to obtain. Self-supervised pretraining methods can be leveraged to enhance **language understanding** or **speech generation capabilities**. 
    
    For example, the **text encoder** in TTS can be enhanced by the **pre-trained BERT models**, and the **speech decoder** in TTS can be **pre-trained** through **autoregressive Mel-spectrogram** **prediction** or jointed trained with the **voice conversion task**. 
    
    Besides, speech can be quantized into the **discrete token sequence** to resemble the **phoneme** or **character sequence**. In this way, the quantized discrete tokens and the speech can be regarded as **pseudo-paired data** to pre-train a TTS model, which is **then fine-tuned** on a few truly paired text and speech data.
    
- **Cross-lingual transfer**.
    
    Although paired text and speech data is scarce in low-resource languages, it is abundant in **rich-resource languages**. Since human languages share similar **vocal organs**, **pronunciations** [389], and **semantic structures** [341], **pre-training the TTS models** in rich-resource languages can help the **mapping between text and speech** in low-resource languages. Usually, there are different **phoneme sets** between rich and low-resource languages. 
    
    Thus, **Chen et al.** [43] propose to map the **embeddings** between the phoneme sets from **different languages.**
    
    **LRSpeech** [396] discards the pre-trained phoneme embeddings and initializes the phoneme embeddings **from scratch for low-resource languages.** 
    
    The **international phonetic alphabet** (IPA) [109] or **byte representation** [108] is adopted to support **arbitrary texts in multiple languages.** Besides, **language similarity** [341] can also be considered when conducting the cross-lingual transfer.
    
- **Cross-speaker transfer**.
    
    When a certain speaker has **limited speech data**, the data from other speakers can be leveraged to improve the synthesis quality of this speaker. This can be achieved by converting the voice of other speakers into this target voice through **voice conversion** to increase the training data, or by adapting the TTS models trained on other voices to this target voice through **voice adaptation** or **voice cloning which** are introduced in Section 3.6.
    
- **Speech chain/Back transformation.**
    
    **Text-to-speech (TTS)** and **automatic speech recognition (ASR)** are two dual tasks [285] and can be leveraged together to improve each other. Techniques like **speech chain** [350, 351] and **back transformation** [291, 396] leverage additional unpaired text and speech data to boost the performance of **TTS and ASR.**
    
- **Dataset mining in the wild.**
    
     In some scenarios, there may exist some **low-quality paired text and speech** data in the Web. Cooper [59], Hu et al. [122] propose to mine this kind of data and develop sophisticated techniques to train a TTS model. Some techniques such as **speech enhancement** [362], **denoising** [434], and **disentangling** [383, 120] can be leveraged to **improve the quality of the speech data** mined in the wild.
    

### 🐬**Robust TTS**

> A good TTS system should be robust to always generate “correct” speech according to text even when encountering corner cases. In neural TTS, robust issues such as **word skipping**, **repeating**, and **attention collapse** often happen in acoustic models when generating a Mel-spectrogram sequence from a character/phoneme sequence.
> 

Basically speaking, the causes of these robust issues are from two categories: 

1) The difficulty in **learning the alignments** between characters/phonemes and Mel-spectrograms; 

2) The **exposure bias** and **error propagation** problems incurred in an autoregressive generation. 

![Untitled](A%20Survey%20on%20Neural%20Speech%20Synthesis%20%E2%80%94%20Brief%2086f5fa136b164a74a031d743774b608d/Untitled%2014.png)

- 🌺**enhancing attention**
    
    ![Untitled](A%20Survey%20on%20Neural%20Speech%20Synthesis%20%E2%80%94%20Brief%2086f5fa136b164a74a031d743774b608d/Untitled%2015.png)
    
     **Local**: one character/phoneme token can be aligned to **one or multiple** consecutive Mel-spectrogram frames, while one Mel-spectrogram frame can only be aligned to a **single** character/phoneme token, which can avoid **blurry attention** and **attention collapse**; 
    
    1. **Monotonic**: if character A is behind character B, the Mel-spectrogram corresponding to A is also behind that corresponding to B, which can avoid **word repeating**; 
    2. **Complete**: each character/phoneme token must be covered by at least one Mel-spectrogram frame, which can avoid **word skipping**
    - **Content-based attention.**
        
        The early attention mechanisms adopted in TTS (e.g. Tacotron) are **content-based**, where the attention distributions are determined by the **degree of match** between the **hidden** **representations** from the **encoder** and **decoder.** Content-based attention is suitable for the tasks such as neural **machine translation** where the alignments between the source and target tokens are **purely based on semantic meaning** (content). However, for the tasks like **automatic speech recognition** and **text-to-speech synthesis**, the alignments between text and speech have some specific properties. For example, in TTS, the attention alignments should be **local**, **monotonic**, and **complete**. Therefore, advanced **attention mechanisms** should be designed to better leverage these properties.
        
    - **Location-based attention**
        
        Considering the alignments between text and speech are depending on their positions, location-based attention [93, 17] is proposed to leverage the positional information for alignment. Several TTS models such as **Char2Wav**, **VoiceLoop**, and **MelNet** adopt location-based attention. location-based attention can ensure the **monotonicity property** if properly handled.
        
    - **Content/Location-based hybrid attention**
        
        To combine the advantages of content and location-based
        attentions, Chorowski et al., Shen et al. introduce **location-sensitive attention**: when calculating the current attention alignment, the **previous attention alignment is used.** In this way, the attention would be more stable due to monotonic alignment.
        
    - **Monotonic attention**
        
        > For monotonic attention, the **attention position** is monotonically increasing, which also leverages the prior that the **alignments between text and speech are monotonic**.
        > 
        > 
        > In this way, it can avoid **skipping** and **repeating** **issues**. However, the **completeness property** cannot be guaranteed in the above monotonic attention. 
        > 
        - Therefore, He et al. propose **stepwise monotonic attention**, where in each decoding step, the attention alignment position moves forward **at most one step** and is not allowed to **skip any input unit**.
    - **Windowing or off-diagonal penalty.**
        - Since attention alignments are monotonic and diagonal, Chorowski et al. [50], Tachibana et al. [332], Zhang et al. [438], Ping et al. [270], Chen et al. [39] propose to restrict the attention on the source sequence into a **window subset**. In this way, the
        **learning flexibility and difficulty is reduced.**
        - Chen et al. [39] use **penalty loss** for **off-diagonal attention weights**, by constructing a **band mask** and encouraging the **attention weights** to be
        distributed in the **diagonal band**
    - **Enhancing encoder-decoder connection.**
        
        > Since speech has **more correlation** among adjacent frames,
        the decoder itself contains enough information to predict the next frame and thus tends to **ignore the text information** from the encoder.
        > 
        - Wang et al. [382], Shen et al. [303] use **multi-frame prediction** that generates **multiple non-overlapping output frame**s at each decoder step. In this way, in order to predict consecutive frames, the decoder is f**orced to leverage information from the encoder side**, which can improve the alignment learning.
        - Other works also use a **large dropout** in the prenet before the decoder [382, 303, 39], or small hidden size in the prenet as a **bottleneck** [39], which can **prevent simply copying the previous speech frame** when predicting the current speech frame. The decoder will get more information from the encoder side, which benefits the alignment learning.
        - Ping et al. [270], Chen et al. [39] propose to enhance the connection of the **positional information** between source and target sequences, which benefits the attention alignment learning.
        - Liu et al. [203] leverage CTC [94] **based ASR as a cycle loss** to encourage the generated Mel-spectrograms to contain **text information**, which can also enhance the encoder-decoder connection for better attention alignment.
    - **Positional attention**
        
        Some **non-autoregressive generation models** [268, 234] leverage **position information** as the **query** to attend to the **key** and **value** from the encoder, which is another way to build the connection between the encoder and the decoder for a parallel generation.
        
- 🥪**Replacing Attention with Duration Prediction**
    
    > While improving the attention alignments between text and speech can alleviate the robust issues to some extent, **it cannot totally avoid them**.
    > 
    
    Thus, some works propose to totally **remove the encoder-decoder attention**, explicitly **predict the duration of each character/phoneme,** and expand the **text hidden sequence** according to the **duration** to **match the length of the Mel-spectrogram sequence.** After that, the model can generate a Mel-spectrogram sequence in an autoregressive or non-autoregressive manner. 
    
    Existing works to investigate the **duration prediction** in neural TTS can be categorized into two perspectives: 
    
    1) Using **external alignment tools** or **joint training** to get the **duration** label.
    
    2) Optimizing the **duration prediction** in an **end-to-end** way or using **ground-truth** duration in **training and predicted duration in inference.**
    
    ![Untitled](A%20Survey%20on%20Neural%20Speech%20Synthesis%20%E2%80%94%20Brief%2086f5fa136b164a74a031d743774b608d/Untitled%2016.png)
    
    - **External alignment**
        1. **Encoder-decoder attention:**
        **FastSpeech** obtains the duration label from the **attention alignments** of an autoregressive acoustic model. **SpeedySpeech** follows a similar pipeline of FastSpeech to extract the duration from an **autoregressive teacher model** but replaces the whole network structure with **purely CNN.**
        2. **CTC alignment:**
            
            Beliaev et al. [19] leverage a **CTC-based ASR model** to provide the alignments between phoneme and Mel-spectrogram sequence. 
            
        3. **HMM alignment:** 
            
            **FastSpeech 2** leverages the **HMM-based Montreal forced alignment (MFA)** to get the duration.  
            
        4. Other works such as **DurIAN**, **RobuTrans**, **Parallel Tacotron**, and **Non-Attentive Tacotron** use **forced alignment or speech recognition tools** to get the alignments.
    - **Internal alignment**
        - **AlignTTS** follows the basic model structure of **FastSpeech** but leverages a dynamic **programming-based method** to learn the alignments between text and Mel-spectrogram sequences with multi-stage training.
        - **JDI-T** follows FastSpeech to extract duration from an
        autoregressive teacher model, but jointly trains the autoregressive and non-autoregressive models,
        which does not need two-stage training.
        - **Glow-TTS** leverages a novel monotonic alignment search to extract duration.
        - **EATS** leverages the **interpolation** and **soft dynamic time warping (DTW) loss** to optimize the duration prediction in a **fully end-to-end way.**
    - **Non-end-to-end optimization**
        
        Typical duration prediction methods usually use duration obtained from external/internal alignment tools for training,
        and use predicted duration for inference. The predicted duration is not end-to-end optimized by receiving guiding signal (gradients) from the mel-spectrogram loss.
        
    - **End-to-end optimization**
        
        In order to jointly optimize the duration to achieve better prosody
        
        - **EATS**
            
            predicts the duration using an **internal module** and optimizes the duration **end-to-end** with the help of **duration** **interpolation** and **soft DTW loss**. 
            
        - **Parallel Tacotron 2**
            
            follows the practice of **EATS** to ensure **differentiable duration prediction**. 
            
        - **Non-Attentive Tacotron**
        proposes **semi-supervised learning** for duration prediction, where the predicted duration can be used for **upsampling** if no duration label is available.
- 💥**Enhancing AR Generation**
    
    <aside>
    💡 Autoregressive sequence generation usually suffers from **exposure bias** and **error propagation**.
    
    </aside>
    
    > **Exposure bias** refers to the sequence generation model that is usually **trained** by taking the **previous ground-truth** value as **input** (i.e., teacher-forcing) but generates the sequence autoregressively by taking the **previous predicted value** as **input** in **inference.**
    > 
    - The mismatch between training and inference can cause **error propagation** in inference, where the **prediction errors** can **accumulate quickly** along the generated sequence.
    
    ---
    
    - Guo et al. leverage **professor forcing** to alleviate the mismatch between the different distributions of real and predicted data.
    - Liu et al. conduct **teacher-student distillation** to reduce the exposure bias problem, where the teacher is trained with **teacher-forcing mode**, and the student takes the previously **predicted value** as input and is optimized to reduce the **distance of hidden states between the teacher and student models**.
    
    ---
    
    <aside>
    💡 Considering the **right part** of the generated Mel-spectrogram sequence is usually **worse** than that in the **left part** due to **error propagation**, some works leverage **both left-to-right and right-to-left generations** for **data augmentation** and **regularization**.
    
    </aside>
    
    - Vainer and Dušek [361] leverage some **data augmentations** to alleviate the exposure bias and error propagation issues, by adding **some random Gaussian noises** to each input spectrogram pixel to simulate the **prediction errors** and **degrading the input spectrograms** by randomly **replacing several frames with random frames** to encourage the model to use temporally more **distant frames**.
- 🦹🏻**Replacing AR Generation with NAR Generation**
    
    > Although the **exposure bias** and **error propagation** problems in AR generation can be alleviated through the above methods, **the problems cannot be addressed thoroughly.**
    > 
    
    Therefore, some works directly adopt **non-autoregressive generation** to avoid these issues.
    
    - Some works such as **ParaNet** and **Flow-TTS** use p**ositional attention** [270] for the **text and speech alignment** in a parallel generation.
    - The remaining works such as **FastSpeech** and **EATS** use **duration prediction** to bridge the length mismatch between text and speech sequences. Based on the introductions in the above subsections, we have a new category of TTS according to the **alignment learning** and **AR/NAR generation**, as shown in Table 13:
    
    ![Untitled](A%20Survey%20on%20Neural%20Speech%20Synthesis%20%E2%80%94%20Brief%2086f5fa136b164a74a031d743774b608d/Untitled%2017.png)
    
    1. **AR + Attention**, such as **Tacotron**, **DeepVoice 3**, and **TransformerTTS.**
    2. **AR + Non-Attention (Duration)**, such as **DurIAN**, **RobuTrans**, and **Non-Attentive Tacotron**. 
    3. **NonAR + Attention**, such as **ParaNet**, **Flow-TTS**, and **VARA-TTS**.
    4. **Non-AR + Non-Attention**, such as **FastSpeech 1/2**, **Glow-TTS**, and **EATS**.

### 🍙**Expressive TTS**

> The goal of **text-to-speech** is to synthesize intelligible and natural speech. The research on expressiveness TTS covers broad topics including **modeling**, **disentangling**, **controlling**, and **transferring** the **content**, **timbre**, **prosody**, **style**, **emotion**, etc. We review those topics in this subsection.
> 
- **A key for expressive TTS:** **one-to-many mapping**
    - **duration, pitch, sound volumn, speaker style, emotion, etc.** are variable
    - modeling one-to-many mapping under the regular L1 loss without enough input information will cause **over-smoothing Mel-spectrogram prediction**
- 🧉**Categorization of Variation Information**
    - **Text Information**
        - Some works improve the representation learning of text through **enhanced word embeddings** or **text pre-training**
    - **Speaker or timbre information**
        - Some multi-speaker TTS systems explicitly model the **speaker representations** through a **speaker lookup table** or **speaker encoder**
    - **Prosody, style, and emotion information**
        - key information to improve the expressiveness of speech
    - **Recording devices or noise environments**
        - affect speech quality
- 🎲**Modeling Variation Information**
    
    ![Untitled](A%20Survey%20on%20Neural%20Speech%20Synthesis%20%E2%80%94%20Brief%2086f5fa136b164a74a031d743774b608d/Untitled%2018.png)
    
    - **Information Type**
        - **Explicit Information:**
            
            For explicit information, we directly use them as input to enhance the models for expressive synthesis. 
            
        1. **Get the language ID, speaker ID, style, and prosody** from the labeling data. For example, the **prosody information** can be labeled according to some **annotation schemas**, such as **ToBI**, **AuToBI**, **Tilt**, **INTSINT**, and **SLAM**. 
        2. **Extract the pitch and energy information** from speech and **extract duration** from paired text and speech data. 
        - **Implicit Information:**
            
            In some situations, there are no explicit labels available, or explicit labeling usually causes much human effort and cannot cover specific or fine-grained variation information. 
            
            - **Reference encoder.**
                - Skerry-Ryan et al. define **prosody as the variation** in speech signals that remains after removing variation due to text content, speaker timbre, and channel effects, and model prosody through a reference encoder, which does not require explicit annotations. Specifically, it **extracts prosody embeddings** from reference audio and uses it as the **input of the decoder**. During training, **ground-truth reference audio** is used, and during inference, **another refer audio is used** to synthesize speech with similar prosody.
                - Wang et al. [383] extract **embeddings from reference audio** and use them as the **query to attend** (through Q/K/V based attention [368]) **banks of style tokens**, and the attention results are used as the prosody condition of TTS models for expressive speech synthesis. The style tokens can increase the **capacity and variation of TTS models** to learn different kinds of styles, and enable **knowledge sharing** across data samples in the dataset. Each token in the style token bank can learn different prosody representations, such as **different speaking rates** and **emotions.** During inference, it can use **reference audio** to attend and extract **prosody representations,** or simply pick **one or some style tokens** to synthesize speech.
            - **Variational autoencoder** [119, 4, 443, 120, 103, 324, 325, 74].
                
                Zhang et al. [443] leverage **VAE** to model the variance information in the latent space with **Gaussian prior as a regularization**, which can enable **expressive modeling** and **control** of synthesized styles. 
                
                Some works [4, 120, 2, 74] also leverage the **VAE** framework to better model the variance information for expressive synthesis.
                
            - **Advanced generative models** [224, 186, 366, 234, 159, 70, 141, 185]. One way to alleviate the **one-to-many mapping problem** and **combat over-smoothing prediction** is to use a**dvanced generative
            models** to implicitly learn the variation information, which can better model the multi-modal distribution.
            - **Text pre-training** [81, 104, 393, 143, 98, 454], can provide better text representations by using **pre-trained word embeddings** or **model parameters.**
    - **Information Granularity**
        
        Variation information can be modeled in different granularities. We describe this information from coarse-grained to fine-grained levels: 
        
        1) **Language level and speaker level** [445, 247, 39], where multilingual and multispeaker TTS systems use language ID or speaker ID to differentiate languages and speakers.
        
        2) **Paragraph level** [11, 395, 376], where a TTS model needs to consider the connections between utterances/sentences for long-form reading. 
        
        3) **Utterance level** [309, 383, 142, 321, 207, 40], where a single hidden vector is extracted from the reference speech to represent the timber/style/prosody of this utterance. 
        
        4) **Word/syllable level** [325, 116, 45, 335], which can model the fine-grained style/prosody information that cannot be covered by utterance level information. 
        
        5) **Character/phoneme level** [188, 324, 430, 325, 45, 40, 189], such as duration, pitch or prosody information. 
        
        6) **Frame level** [188, 158, 49, 434], the most fine-grained information. Some corresponding works on different granularities can be found in Table 14. Furthermore, modeling the variance information with hierarchical structure that covers different granularities is helpful for expressive synthesis. Suni et al. [330] demonstrate that hierarchical structures of prosody intrinsically exist in spoken languages. Kenter et al. [158] predict prosody features from frame and phoneme levels to syllable level, and concatenate with word- and sentencelevel features. Hono et al. [116] leverage a multi-grained VAE to obtain different time-resolution latent variables and sample finer-level latent variables from coarser-level ones (e.g., from utterance level to phrase level and then to word level). Sun et al. [325] use VAE to model variance information on both phoneme and word levels and combine them together to feed into the decoder. Chien and Lee [45] study on prosody prediction and propose a hierarchical structure from the word to phoneme level to improve the prosody prediction.
        
- 🔐**Disentangling, Controlling, and Transferring**
    
    ![Untitled](A%20Survey%20on%20Neural%20Speech%20Synthesis%20%E2%80%94%20Brief%2086f5fa136b164a74a031d743774b608d/Untitled%2019.png)
    

### 🍩**Adaptive TTS**

![Untitled](A%20Survey%20on%20Neural%20Speech%20Synthesis%20%E2%80%94%20Brief%2086f5fa136b164a74a031d743774b608d/Untitled%2020.png)

- General Adaptation
    
    Source Model Gerneralization
    
    Cross-Domain Adaptation
    
- Efficient Adaptation
    
    Few data adaptation
    
    Few parameter adaptation
    
    Untranscribed data adaptation
    
    Zero-shot adaptation
    

## 🤺Resources

### ✈️**Open-Source Implementations**

ESPnet-TTS [https://github.com/espnet/espnet](https://github.com/espnet/espnet)
Mozilla-TTS [https://github.com/mozilla/TTS](https://github.com/mozilla/TTS)
TensorflowTTS [https://github.com/TensorSpeech/TensorflowTTS](https://github.com/TensorSpeech/TensorflowTTS)
Coqui-TTS [https://github.com/coqui-ai/TTS](https://github.com/coqui-ai/TTS)
Parakeet [https://github.com/PaddlePaddle/Parakeet](https://github.com/PaddlePaddle/Parakeet)
NeMo [https://github.com/NVIDIA/NeMo](https://github.com/NVIDIA/NeMo)
WaveNet [https://github.com/ibab/tensorflow-wavenet](https://github.com/ibab/tensorflow-wavenet)
WaveNet [https://github.com/r9y9/wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder)
WaveNet [https://github.com/basveeling/wavenet](https://github.com/basveeling/wavenet)
SampleRNN [https://github.com/soroushmehr/sampleRNN_ICLR2017](https://github.com/soroushmehr/sampleRNN_ICLR2017)
Char2Wav [https://github.com/sotelo/parrot](https://github.com/sotelo/parrot)
Tacotron [https://github.com/keithito/tacotron](https://github.com/keithito/tacotron)
Tacotron [https://github.com/Kyubyong/tacotron](https://github.com/Kyubyong/tacotron)
Tacotron 2 [https://github.com/Rayhane-mamah/Tacotron-2](https://github.com/Rayhane-mamah/Tacotron-2)
Tacotron 2 [https://github.com/NVIDIA/tacotron2](https://github.com/NVIDIA/tacotron2)
DeepVoice 3 [https://github.com/r9y9/deepvoice3_pytorch](https://github.com/r9y9/deepvoice3_pytorch)
TransformerTTS [https://github.com/as-ideas/TransformerTTS](https://github.com/as-ideas/TransformerTTS)
FastSpeech [https://github.com/xcmyz/FastSpeech](https://github.com/xcmyz/FastSpeech)
FastSpeech 2 [https://github.com/ming024/FastSpeech2](https://github.com/ming024/FastSpeech2)
MelGAN [https://github.com/descriptinc/melgan-neurips](https://github.com/descriptinc/melgan-neurips)
MelGAN [https://github.com/seungwonpark/melgan](https://github.com/seungwonpark/melgan)
WaveRNN [https://github.com/fatchord/WaveRNN](https://github.com/fatchord/WaveRNN)
LPCNet [https://github.com/mozilla/LPCNet](https://github.com/mozilla/LPCNet)
WaveGlow [https://github.com/NVIDIA/WaveGlow](https://github.com/NVIDIA/WaveGlow)
FloWaveNet [https://github.com/ksw0306/FloWaveNet](https://github.com/ksw0306/FloWaveNet)
WaveGAN [https://github.com/chrisdonahue/wavegan](https://github.com/chrisdonahue/wavegan)
GAN-TTS [https://github.com/r9y9/gantts](https://github.com/r9y9/gantts)
Parallel WaveGAN [https://github.com/kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)
HiFi-GAN [https://github.com/jik876/hifi-gan](https://github.com/jik876/hifi-gan)
Glow-TTS [https://github.com/jaywalnut310/glow-tts](https://github.com/jaywalnut310/glow-tts)
Flowtron [https://github.com/NVIDIA/flowtron](https://github.com/NVIDIA/flowtron)
DiffWave [https://github.com/lmnt-com/diffwave](https://github.com/lmnt-com/diffwave)
WaveGrad [https://github.com/ivanvovk/WaveGrad](https://github.com/ivanvovk/WaveGrad)
VITS [https://github.com/jaywalnut310/vits](https://github.com/jaywalnut310/vits) 

TTS Samples [https://github.com/seungwonpark/awesome-tts-samples](https://github.com/seungwonpark/awesome-tts-samples)
Software/Tool for Audio [https://github.com/faroit/awesome-python-scientific-audio](https://github.com/faroit/awesome-python-scientific-audio)

### **TTS Tutorials & Keynotes**

TTS Tutorial at ISCSLP 2014 [https://www.superlectures.com/iscslp2014/tutorial-4-deep-learning-for-sp](https://www.superlectures.com/iscslp2014/tutorial-4-deep-learning-for-sp)
eech-generation-and-synthesis
TTS Tutorial at ISCSLP 2016 [http://staff.ustc.edu.cn/~zhling/download/ISCSLP16_tutorial_DLSPSS.pdf](http://staff.ustc.edu.cn/~zhling/download/ISCSLP16_tutorial_DLSPSS.pdf)
TTS Tutorial at IEICE [https://www.slideshare.net/jyamagis/tutorial-on-endtoend-texttospeech-sy](https://www.slideshare.net/jyamagis/tutorial-on-endtoend-texttospeech-sy)
nthesis-part-1-neural-waveform-modeling
Generative Models for Speech [https://www.youtube.com/watch?v=vEAq_sBf1CA](https://www.youtube.com/watch?v=vEAq_sBf1CA)
Generative Model-Based TTS [https://static.googleusercontent.com/media/research.google.com/en//pubs/](https://static.googleusercontent.com/media/research.google.com/en//pubs/)
archive/45882.pdf
Keynote at INTERSPEECH [http://www.sp.nitech.ac.jp/~tokuda/INTERSPEECH2019.pdf](http://www.sp.nitech.ac.jp/~tokuda/INTERSPEECH2019.pdf)
TTS Tutorial at ISCSLP 2021 [https://www.microsoft.com/en-us/research/uploads/prod/2021/02/ISCSLP2021](https://www.microsoft.com/en-us/research/uploads/prod/2021/02/ISCSLP2021)
-TTS-Tutorial.pdf
TTS Webinar [https://www.youtube.com/watch?v=MA8PCvmr8B0](https://www.youtube.com/watch?v=MA8PCvmr8B0)
TTS Tutorial at IJCAI 2021 [https://tts-tutorial.github.io/ijcai2021/](https://tts-tutorial.github.io/ijcai2021/)

### **TTS Challenges**

Blizzard Challenge [http://www.festvox.org/blizzard/](http://www.festvox.org/blizzard/)
Zero Resource Speech Challenge [https://www.zerospeech.com/](https://www.zerospeech.com/)
ICASSP2021 M2VoC [http://challenge.ai.iqiyi.com/detail?raceId=5fb2688224954e0b48431fe0](http://challenge.ai.iqiyi.com/detail?raceId=5fb2688224954e0b48431fe0)
Voice Conversion Challenge [http://www.vc-challenge.org/](http://www.vc-challenge.org/)

### **TTS Corpora**

![Untitled](A%20Survey%20on%20Neural%20Speech%20Synthesis%20%E2%80%94%20Brief%2086f5fa136b164a74a031d743774b608d/Untitled%2021.png)

## 🍛**Future Directions**

![Untitled](A%20Survey%20on%20Neural%20Speech%20Synthesis%20%E2%80%94%20Brief%2086f5fa136b164a74a031d743774b608d/Untitled%2022.png)

![Untitled](A%20Survey%20on%20Neural%20Speech%20Synthesis%20%E2%80%94%20Brief%2086f5fa136b164a74a031d743774b608d/Untitled%2023.png)

![Untitled](A%20Survey%20on%20Neural%20Speech%20Synthesis%20%E2%80%94%20Brief%2086f5fa136b164a74a031d743774b608d/Untitled%2024.png)

### **High-quality speech synthesis**

### **Efficient speech synthesis**

### voice conversion

### singing voice synthesis

### talking face synthesis