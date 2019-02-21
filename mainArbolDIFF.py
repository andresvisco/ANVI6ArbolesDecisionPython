import sklearn

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydot
import pydotplus
from sklearn import tree
import graphviz
import numpy as np
import os     

os.environ["PATH"] += os.pathsep + 'C://Program Files (x86)//Graphviz2.38//bin'


datos = pd.read_csv('DataSets/GenelbaFichadasArbolesDIFFAREARetraso.csv')
feature_cols = ['DIFFAgrupadoID','AREA_ID','DIA_ID']
df = pd.DataFrame(data=datos, columns=['DIFFAgrupadoID','AREA_ID','DIA_ID'])
y=df['DIFFAgrupadoID']

print("Target: ", y)
print("*"*60)
print("DF: ", df)


dtree=DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=4)
dtree.fit(df, y)

# dot_data = StringIO()
dot_data = tree.export_graphviz(dtree, out_file=None,filled=True,rounded=True,feature_names=feature_cols)
# graph = pydot.graph_from_dot_data(dot_data.getvalue())
# graph = graphviz.Source(dot_data)
# print(graph)
# graph.render()

graph = pydotplus.graph_from_dot_data(dot_data)
nodes = graph.get_node_list()
diffVal=0
for node in nodes:
    if node.get_label():
        values = node.get_label().split("\\n")
        new_text_values = []
        for v in values:
            new_text_values.append(v)
            if v.__contains__('DIFFAgrupadoID <= '):
                pf = v.replace('\"DIFFAgrupadoID <= ', '')
                diffVal = float(pf)

            if v.__contains__('samples'):
                break
        nt = "\n".join(new_text_values)
        node.set_label(nt)

        if diffVal<-60:
            node.set_fillcolor("red")
        elif diffVal<-30:
            node.set_fillcolor("orange")
        elif diffVal<-15:
            node.set_fillcolor("yellow")
        elif diffVal<-1:
            node.set_fillcolor("blue")
        else:
            node.set_fillcolor("forestgreen")
graph.write_png('colored_treeARBOLAREARetraso.png')