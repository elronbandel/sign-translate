from transformers import pipeline

print("Loading model")
pipe = pipeline('translation', 'outputs')

print("Translating")
for text in ["string", "advanced", "raw"]:
    print(text, pipe('en ' + text)[0]['translation_text'])
