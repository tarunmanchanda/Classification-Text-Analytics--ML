#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 15:37:00 2019

@author: Preeti, Sofana, Tarun
"""

# unpickling the classifier and vectorizer
import pickle

with open('classifier.pickle','rb') as f:
    clf2 = pickle.load(f)
with open('bowmodel.pickle','rb') as f:
    bow = pickle.load(f)
actualLabelClass = 2
# Testing our classifier and model
sample = ["followed away went house empty gate told enough civilly way asked like beast wild house live havin guv ye pity madhouse bloomin said mind sir yer bless lor saying got place kind mind making place looking contented notice man signed window opened swing hinder would said murder wanting robbing accused man whereon beggar mouthed foul shut telling contented enough fellow decent seemed man tongue lay could names foul called within rate began patient room renfield window passed house come one saw dinner smoke window study looking strangers way porter ask gate stopped men away ran twice patient remember house abut grounds whose house empty call made men two cart carrier afternoon results unhappy unattended happened fortunately ending dreadful might outbreak another say renfield patient regard charge left everything conditions report enclose wishes accordance sir dear september seward john etc etc c k r hennessey patrick report harker mina blessings lucy"]
sample = bow.transform(sample).toarray()
predictedLabelClass = clf2.predict(sample)
if( predictedLabelClass == actualLabelClass ):
    print("Test passed")
else:
    print("Test failed")
