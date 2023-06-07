import matplotlib.pyplot as plt
from data_loader import load_training, load_testing, load_validation, load_meta
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

LABEL_NAMES = load_meta()

tr_X = load_training()[0]
tr_y = load_training()[1]
te_X = load_testing()[0]
te_y = load_testing()[1]
val_X = load_validation()[0]
val_y = load_validation()[1]

k_list = [1, 5, 10, 25, 50]

tr_errors = []
te_errors = []

print("running...")
for K in k_list:
    print(f"...{K}")
    clf = KNeighborsClassifier(n_neighbors=K)
    clf.fit(tr_X, tr_y)
    # Predict on data
    pred_tr_y = clf.predict(tr_X)
    pred_te_y = clf.predict(te_X)

    # Calculate accuracy
    tr_acc = accuracy_score(tr_y, pred_tr_y)
    tr_errors.append(1 - tr_acc)

    te_acc = accuracy_score(te_y, pred_te_y)
    te_errors.append(1 - te_acc)

plt.title("Varying error rates on k different neighbors")
plt.plot(k_list, tr_errors, c="red", label="training")
plt.plot(k_list, te_errors, c="green", label="testing")
plt.legend()
plt.xlabel("k neighbors")
plt.ylabel("error rate")
plt.show()

print(tr_errors)
print(te_errors)

# [0.0, 0.5021249999999999, 0.578875, 0.638525, 0.6685749999999999] <= training err
# [0.6642, 0.665, 0.6662, 0.6754, 0.6817] <= testing err

# val_errors = []
#
# for K in k_list:
#     print(f"...{K}")
#     clf = KNeighborsClassifier(n_neighbors=K)
#     clf.fit(tr_X, tr_y)
#     # Predict on data
#     pred_val_y = clf.predict(val_X)
#
#     # Calculate accuracy
#     val_acc = accuracy_score(val_y, pred_val_y)
#     val_errors.append(1 - val_acc)
#
# plt.title("Varying error rates on k different neighbors")
# plt.plot(k_list, val_errors, c="red", label="Validation")
# plt.legend()
# plt.xlabel("k neighbors")
# plt.ylabel("error rate")
# plt.show()
#
# print(val_errors)
# [0.6591, 0.6692, 0.6671, 0.6666000000000001, 0.683]
