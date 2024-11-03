# -*- coding: utf-8 -*-
"""
Created on Sat May 11 14:46:39 2019

@author: Maram
"""

from sklearn.metrics import classification_report as cr
from sklearn.metrics import accuracy_score as acc
import os
from itertools import groupby
from operator import itemgetter
from collections import defaultdict
import numpy as np
import argparse

MAIN_THRESHOLDS = [5, 10, 15, 20]

def loadTrueLabels(task,inputLabelsFile):
    trueLabelsWIds = {}
    trueLabels = []
                
    with open(inputLabelsFile, encoding='utf-8') as f:
        for line in f:
            splits = line.split(",")
            pId = splits[len(splits)-2]
            label = splits[len(splits)-1].strip()
            trueLabelsWIds[pId] = label

    #ensure the labels are stored in a sorted order by page/paragraph ID      
    for pId,label in sorted(trueLabelsWIds.items()):
        trueLabels.append(label)

    return trueLabels
    

def loadTrueLabelsPerClaim(task,inputLabelsFile):
    trueLabelsWIds = {}
    trueLabels = []
    trueLabelsTuples = []
    claimSet = set([])
    with open(inputLabelsFile, encoding='utf-8') as f:
        for line in f:
            splits = line.split(",")
            pId = splits[len(splits)-2]
            label = splits[len(splits)-1].strip()
            trueLabelsWIds[pId] = label
            claimSet.add(pId[:-3])

    
    if task != "A":
        #ensure the labels are stored in a sorted order by page/paragraph ID      
        for pId,label in sorted(trueLabelsWIds.items()):
            if task == "C":
                cId = pId[:-6]
            else:
                cId = pId[:-3]
            trueLabelsTuples.append((cId,label))
        
        trueLabels = [(k, list(list(zip(*g))[1])) for k, g in groupby(trueLabelsTuples, itemgetter(0))]
        
    else:
        for cId in claimSet:
            perClaimDic = {}
            result = [(key, value) for key, value in trueLabelsWIds.items() if key.startswith(cId)]
            for key, value in result:
                perClaimDic[key] = int(value)
            trueLabels.append((cId,perClaimDic))
    
    return trueLabels


def editResultFormat2ClassOverallCB(task,classificationResults,accuracy):
    resultsOutput = ""
    results = classificationResults['Useful']
    del results['support']
    for metric,value in results.items():
        resultsOutput += str.capitalize(metric)+": "+ str(round(value,2)) + "\n"
    
    resultsOutput = resultsOutput.replace("F1-score","F1")
    resultsOutput += "Accuracy: " + str(round(accuracy,2)) + "\n"
    
    return resultsOutput

def editResultFormat2ClassPerClaimCB(task,classificationResults,accuracy):
    data = []
    header = ['Claim','Precision','Recall','F1','Accuracy']
    data.append(header)
    allvalues = defaultdict(list)
    resultsOutput = ""
    for claim,res in classificationResults.items():
        perClaimValues = [claim]
        acc = accuracy[claim]
        results = res['Useful']
        del results['support']
        for metric,value in results.items():
            allvalues[metric].append(value)
            perClaimValues.append(str(round(value,2)))
        perClaimValues.append(str(round(acc,2)))
        allvalues["acc"].append(acc)
        data.append(perClaimValues)
    
    avg = ["Average"]
    for key, value in allvalues.items():
        avg.append(str(round((sum(value)/len(value)),2)))
    data.append(avg)
    
    col_width = max(len(str(word)) for row in data for word in row) + 2  # padding
    for row in data:
        rowToWrite = "".join(str(word).ljust(col_width) for word in row)
        resultsOutput += rowToWrite + "\n"
    
    return resultsOutput

def editResultFormat2ClassPerClaimA(task,precisionsPerClaim, ndcgsPerClaim,avg_precisionPerClaim):
    data = []
    header = ['Claim','P@5', 'P@10','P@15','P@20','AP', 'NDCG@5', 'NDCG@10','NDCG@15','NDCG@20']
    data.append(header)
    allvalues = defaultdict(list)
    resultsOutput = ""
    for claim,res in sorted(precisionsPerClaim.items()):
        perClaimValues = [claim]
        ap = avg_precisionPerClaim[claim]
        ndcgs = ndcgsPerClaim[claim]
        i = 1
        for value in res:
            allvalues[header[i]].append(value)
            perClaimValues.append(str(round(value,2)))
            i += 1
            
        perClaimValues.append(str(round(ap,2)))
        allvalues["ap"].append(ap)
        
        for value in ndcgs:
            allvalues[header[i]].append(value)
            perClaimValues.append(str(round(value,2)))
            i += 1
            
        data.append(perClaimValues)
    
    avg = ["Average"]
    for key, value in allvalues.items():
        avg.append(str(round((sum(value)/len(value)),2)))
    data.append(avg)
    
    col_width = max(len(str(word)) for row in data for word in row) + 2  # padding
    for row in data:
        rowToWrite = "".join(str(word).ljust(col_width) for word in row)
        resultsOutput += rowToWrite + "\n"
    
    return resultsOutput


def editResultFormatPerClass(task,classificationResults,accuracy):
    data = []
    header = ['Class','Precision','Recall','F1']
    data.append(header)
    resultsOutput = ""
    for cls,results in classificationResults.items():
        if "micro" in cls or "weighted" in cls: continue
        if "macro" in cls: cls = "Average"
        perClassValues = []
        del results['support']
        perClassValues.append(cls)
        for metric,value in results.items():
            perClassValues.append(str(round(value,2)))
            
        data.append(perClassValues)
    
    col_width = max(len(str(word)) for row in data for word in row) + 2  # padding
    for row in data:
        rowToWrite = "".join(str(word).ljust(col_width) for word in row)
        resultsOutput += rowToWrite + "\n"

    resultsOutput += "\nAccuracy: " + str(round(accuracy,2))
    
    return resultsOutput
    

def writeClassTaskResults(resultsOutputFolder,task,runId,classificationResults,accuracy
                          ,classificationResultsB,accuracyB):
    outDirName = resultsOutputFolder+task+"/"
    if not os.path.exists(outDirName):
        os.makedirs(outDirName)
    
    outResultsFile = outDirName + task + "_" + runId + ".txt"

    if task == "C":
        resultsOutput = "=============================== 2 classes over all claims results ===============================\n"
        resultsOutput += editResultFormat2ClassOverallCB(task,classificationResults,accuracy)
    elif task == "B":
        resultsOutput = "=============================== 2 classes over all claims results ===============================\n"
        resultsOutput += editResultFormat2ClassOverallCB(task,classificationResultsB,accuracyB)
        resultsOutput += "\n=============================== 4 classes over all claims results ===============================\n"
        resultsOutput +=  editResultFormatPerClass(task,classificationResults,accuracy)
    elif task == "D1" or task == "D2":
        resultsOutput = "=============================== 2 classes over all claims results ===============================\n"
        resultsOutput +=  editResultFormatPerClass(task,classificationResults,accuracy)
    with open(outResultsFile, 'a', encoding='utf-8') as f:
        f.write(str(resultsOutput))
        f.write("\n")

def writeClassTaskResultsPerClaim(resultsOutputFolder,task,runId,classificationResults,accuracy):
    outDirName = resultsOutputFolder+task+"/"
    if not os.path.exists(outDirName):
        os.makedirs(outDirName)
    
    outResultsFile = outDirName + task + "_" + runId + ".txt"

    resultsOutput = "================================= 2 classes per claim results =================================\n"
    resultsOutput += editResultFormat2ClassPerClaimCB(task,classificationResults,accuracy)
    with open(outResultsFile, 'w', encoding='utf-8') as f:
        f.write(str(resultsOutput))
        f.write("\n")

def writeClassTaskAResultsPerClaim(resultsOutputFolder,task,runId,precisionsPerClaim, ndcgsPerClaim, avg_precisionPerClaim):
    outDirName = resultsOutputFolder+task+"/"
    if not os.path.exists(outDirName):
        os.makedirs(outDirName)
    
    outResultsFile = outDirName + task + "_" + runId + ".txt"

    resultsOutput = ""
    resultsOutput += editResultFormat2ClassPerClaimA(task,precisionsPerClaim, ndcgsPerClaim, avg_precisionPerClaim)
    with open(outResultsFile, 'w', encoding='utf-8') as f:
        f.write(str(resultsOutput))
        f.write("\n")


def _compute_average_precision(gold_labels, ranked_p):
    """ Computes Average Precision. """

    precisions = []
    num_correct = 0
    num_positive = sum([1 if v >= 1 else 0 for k, v in gold_labels.items()])

    for i, pId in enumerate(ranked_p):
        if gold_labels[pId] >= 1:
            num_correct += 1
            precisions.append(num_correct / (i + 1))
    if precisions:
        avg_prec = sum(precisions) / num_positive
    else:
        avg_prec = 0.0

    return avg_prec

def _compute_precisions(gold_labels, ranked_p, threshold):
    """ Computes Precision at each pId in the ordered list. """
    precision = 0.0
    threshold = min(threshold, len(ranked_p))

    for i, pId in enumerate(ranked_p[:threshold]):
        if gold_labels[pId] >= 1:
            precision += 1.0

    precision /= threshold
    return precision

def dcg_at_k(r, k, method=0):
    """
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max



def evaluateRunsTaskAPerClaim(task,fname,trueLabels):
    predictedLabelsWIds = {}
    predictedLabels = []
    claimSet = set([])
    with open(fname, encoding='utf-8') as f:
        for line in f:
            #to handle badly formated runs
            if "pageID" in line: continue
            if "," in line:
                line = line.replace(",","\t")
                
            splits = line.split("\t")
            pId = splits[len(splits)-3]
            claimSet.add(pId[:-3])
            label = float(splits[len(splits)-4])
            runId = splits[len(splits)-1].strip()

            predictedLabelsWIds[pId] = label
        
        for cId in claimSet:
            perClaimResults = [(key, value) for key, value in predictedLabelsWIds.items() if key.startswith(cId)]
            predictedLabels.append((cId,perClaimResults))
    
    precisionsPerClaim = {}
    ndcgsPerClaim = {}
    avg_precisionPerClaim = {}
    
    for claimTrue,claimPredicted in zip(trueLabels, predictedLabels):
        cId = claimTrue[0]
        truelabels = claimTrue[1]
        predictedlabels = claimPredicted[1]
        ranked_p = [t[0] for t in sorted(predictedlabels, key=lambda x: x[1], reverse=False)]
        # Debugging prints
        ranked_p_labels = [truelabels[p] if truelabels[p]>=0 else 0 for p in ranked_p]
        # Calculate Metrics
        precisions = []
        ndcgs = []
        for thr in MAIN_THRESHOLDS:
            precisions.append(_compute_precisions(truelabels, ranked_p, thr))
            ndcgs.append(ndcg_at_k(ranked_p_labels, thr))
            
        avg_precision = _compute_average_precision(truelabels, ranked_p)
        
        precisionsPerClaim[cId] = precisions
        ndcgsPerClaim[cId] = ndcgs
        avg_precisionPerClaim[cId] = avg_precision

    return runId, precisionsPerClaim, ndcgsPerClaim, avg_precisionPerClaim

   
def evaluateRuns(task,fname,trueLabels):
    predictedLabelsWIds = {}
    predictedLabels = []
    with open(fname, encoding='utf-8') as f:
        for line in f:
            if "pageID" in line: continue
            if "," in line:
                line = line.replace(",","\t")
            splits = line.split("\t")
            pId = splits[len(splits)-3]
            label = splits[len(splits)-2]
            runId = splits[len(splits)-1].strip()

            predictedLabelsWIds[pId] = label
        
        for pId,label in sorted(predictedLabelsWIds.items()):
            predictedLabels.append(label)
    
    target_names = []
    labels = []
    if task == "C":
        target_names = ['Not Useful', 'Useful']
        labels = ['0', '1']
    elif task == "D1" or task == "D2":
        target_names = ['FALSE', 'TRUE']
        labels = ['FALSE', 'TRUE']
    else:
        target_names = ['Not Relevant', 'Not Useful', 'Useful', 'Very Useful']
        labels = ['-1', '0', '1', '2']

    classificationResults = cr(trueLabels, predictedLabels, labels=labels,
                               target_names=target_names,output_dict=True)
    accuracy = acc(trueLabels, predictedLabels)
    
    classificationResultsB = {}
    accuracyB = 0
    
    if task == "B":
        predictedLabels = [l.replace('-1', '0') for l in predictedLabels]
        predictedLabels = [l.replace('2', '1') for l in predictedLabels]
        trueLabels = [l.replace('-1', '0') for l in trueLabels]
        trueLabels = [l.replace('2', '1') for l in trueLabels]
        target_names = ['Not Useful', 'Useful']
        labels = ['0', '1']
        classificationResultsB = cr(trueLabels, predictedLabels, labels=labels,
                                    target_names=target_names,output_dict=True)
        accuracyB = acc(trueLabels, predictedLabels)
    
    return runId,classificationResults,accuracy,classificationResultsB,accuracyB
        
def evaluateRunsPerClaim(task,fname,trueLabels):
    predictedLabelsWIds = {}
    predictedLabels = []
    predictedLabelsTuples = []
    with open(fname, encoding='utf-8') as f:
        for line in f:
            #to handle badly formated runs
            if "pageID" in line: continue
            if "," in line:
                line = line.replace(",","\t")
                
            splits = line.split("\t")
            pId = splits[len(splits)-3]
            label = splits[len(splits)-2]
            runId = splits[len(splits)-1].strip()

            predictedLabelsWIds[pId] = label
        
        for pId,label in sorted(predictedLabelsWIds.items()):
            if task == "C":
                cId = pId[:-6]
            else:
                cId = pId[:-3]
            predictedLabelsTuples.append((cId,label))
    
    predictedLabels = [(k, list(list(zip(*g))[1])) for k, g in groupby(predictedLabelsTuples, itemgetter(0))]

    target_names = ['Not Useful', 'Useful']
    labels = ['0', '1']
    
    if task == "B":
        fixedPredictedLabels = []
        fixedTrueLabels = []
        for cid,l in predictedLabels:
            l2class = ["0" if int(x)<1 else "1" for x in l]
            fixedPredictedLabels.append((cid,l2class))
        for cid,l in trueLabels:
            l2class = ["0" if int(x)<1 else "1" for x in l]
            fixedTrueLabels.append((cid,l2class))
        
        predictedLabels = fixedPredictedLabels
        trueLabels = fixedTrueLabels
        
    classificationResults = {}
    accuracy = {}
    
    for claimTrue,claimPredicted in zip(trueLabels, predictedLabels):
        cId = claimTrue[0]
        truelabels = claimTrue[1]
        predictedlabels = claimPredicted[1]
        results = cr(truelabels, predictedlabels, labels=labels,
                               target_names=target_names,output_dict=True)
        classificationResults[cId] = results
        
        accuracy[cId] = acc(truelabels, predictedlabels,)
   
    return runId,classificationResults,accuracy

            
if __name__ == '__main__':
    #Example command
    #python eval_task2_per_run.py --subtask D2 --gold_file_path D:/LabelsFolder/CT19-T2-Test-SubD-Labels.csv --pred_file_path D:/runFolderPath/run1.csv,D:/runFolderPath/run2.csv --output_folder_path D:/outputFolder/
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subtask",
        help="Single string indicating the subtask the runs your evaluating belong to: [A,B,C,D1,D2]",
        type=str,
        required=True
    )
    parser.add_argument(
        "--gold_file_path",
        help="Single string containing the full path of the file with gold labels.",
        type=str,
        required=True
    )
    parser.add_argument(
        "--pred_file_path",
        help="Single string containing a comma separated list of paths of runs to evaluate.",
        type=str,
        required=True
    )
    parser.add_argument(
        "--output_folder_path",
        help="Single string containing the folder path in which evaluation output will be stored.",
        type=str,
        required=True
    )
    args = parser.parse_args()

    gold_file = args.gold_file_path.strip()
    pred_files = [pred_file.strip() for pred_file in args.pred_file_path.split(",")]
    task = args.subtask.strip()
    resultsOutputFolder = args.output_folder_path.strip()
    
    trueLabels = loadTrueLabels(task,gold_file)
        
    if task != "D1" or task != "D2":
        trueLabelsPC = loadTrueLabelsPerClaim(task,gold_file)

    for fname in pred_files:
        print("Evaluating run: " + fname)
        if task != "A":
            #evaluate and write results for all subtasks except the ranking task, task A
            #evaluate the runs over all claims
            runId,classificationResults,accuracy,classificationResultsB,accuracyB = evaluateRuns(task,fname, trueLabels)
            
            if task == "C" or task == "B":
                #evaluate the runs per claim
                runIdPC,classificationResultsPC,accuracyPC= evaluateRunsPerClaim(task,fname, trueLabelsPC)
                #write the per claim results
                writeClassTaskResultsPerClaim(resultsOutputFolder,task,runIdPC,classificationResultsPC,accuracyPC)
            
            #write overall results 
            writeClassTaskResults(resultsOutputFolder,task,runId,classificationResults,accuracy,
                              classificationResultsB,accuracyB)
        else:
            #evaluate and write results for subtask A
            runId, precisionsPerClaim, ndcgsPerClaim, avg_precisionPerClaim = evaluateRunsTaskAPerClaim(task,fname, trueLabelsPC)
            writeClassTaskAResultsPerClaim(resultsOutputFolder,task,runId,precisionsPerClaim, ndcgsPerClaim, avg_precisionPerClaim)

       