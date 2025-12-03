import math
def summarize(results,metrics=['acc','ppl']):
    outputs = {}
    if len(results) == 0:
        return outputs
    if 'acc' in metrics:
        counts = [r['metrics']['acc'] for r in results if 'acc' in r['metrics']]
        if len(counts)>0:
            outputs['acc'] = sum(counts)/len(counts)
    if 'ppl' in metrics:
        nlls = []
        tokens = 0
        for r in results:
            if 'nll' in r['metrics']:
                nlls.append(r['metrics']['nll'])
                tokens += r['metrics']['tokens']
        if tokens>0:
            outputs['ppl'] = math.exp(sum(nlls)/tokens)
    return outputs


