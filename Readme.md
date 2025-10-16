Use openSMILE (Python opensmile) if you need custom labels and minimal plumbing. Use SpeechBrain SER only if your task≈emotions and you want a pretrained zero-code classifier.

Decision rule

Custo+ sklearn.
m targets or unknown label set → opensmile 
Emotion labels (angry, sad, etc.) and no training → SpeechBrain.