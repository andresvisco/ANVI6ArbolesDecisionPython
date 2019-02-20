import sklearn
from sklearn.datasets import load_iris

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

iris = load_iris()
data1 = pd.DataFrame(data=iris['target'],columns=['target'])
data1.to_csv('datairis.csv')


datos = pd.read_csv('GenelbaFichadasArbolesDIFF.csv')
feature_cols = ['DIFFMinutos','DocumentNumber']
df = pd.DataFrame(data=datos, columns=['DIFFMinutos','DocumentNumber'])
y=df['DIFFMinutos']
AA="AA"
print("Target: ", y)
print("*"*60)
print("DF: ", df)


dtree=DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=5)
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
            if v.__contains__('DIFFMinutos <= '):
                pf = v.replace('\"DIFFMinutos <= ', '')
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

graph.write_png('colored_tree.png')
