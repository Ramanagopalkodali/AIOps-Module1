# Question 1 -- Conceptual: Technical Debt Diagnosis

## 1. Identify the hidden-technical-debt category for each case:

### (a) Entanglement (CACE)

Changing the **"estimated delivery time"** feature unexpectedly affects the unrelated **"favorite restaurants"** feature, showing that changes in one part of the ML system affect other parts.

### (b) Undeclared Consumers

The marketing dashboard team silently consumes the model's raw output table without the ML team's knowledge, creating an invisible dependency.

### (c) Configuration & Glue-Code Debt

The training pipeline consists of **14 undocumented shell scripts** with no orchestration tool, creating tangled and poorly managed pipeline glue code.

## 2. Mitigation for (c)

We can use **DVC pipelines** to replace the chain of undocumented shell scripts with explicitly defined pipeline stages and dependencies.

Each stage can specify its **inputs, outputs, and commands**, making the execution order and dependencies reproducible.

For example, the workflow can be organized as:

**Data validation => Preprocessing => Training => Evaluation**

This reduces the tangled **"pipeline jungle"**, makes the workflow easier to understand and reproduce, and reduces undocumented manual steps.
