目标

测试vllm_quest类，vllm类的运行结果。

其中，vllm_quest类可以选择启用或禁用quest方法，禁用quest方法时相当于一个重新用Pytorch实现了后台attention计算的vllm离线调用。(核心代码在scripts/pred/quest_pytorch/quest_patch.py里）

vllm类是用vllm离线调用实现的。



使用：

创建model文件夹，创建model/llama3-8b-instruct-262k，存放本地模型

运行run_serial_base.sh，得到vllm_quest类（false）在RULER上的测评结果

运行run_serial_vllm.sh，得到vllm类在RULER上的测评结果



结果：

统计了原始RULER仓库中hf framework的结果，以及vllm类，vllm_quest类（False）的结果

模型:llama3-8b-instruct-262k 

seq_length: 131072

其他超参数和RULER仓库默认设置相同

每个task选用前100个sample

|	| budget_size | chunk_size | niah_multikey_1 | niah_multikey_2 | niah_multikey_3	| niah_single_1	| niah_single_2	| niah_single_3	| cwe	| fwe	| niah_multiquery	| niah_multivalue	| qa_1	| qa_2 |	vt	| avg |
| ------ | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| base_hf	| /	| /	| 98.4 | 98.2	| 67.4 | 100 | 99.8	| 99.8 | 0.76	| 70.8 | 91.6	| 89.1 | 40	| 23 | 81.04 | 73.83 |
| base_vllm	| /	| /	| 87.71	| 90.29	| 74.89	| 87.71	| 84.86	| 86.36	| 64.92	| 89.33	| 70.14	| 79.43	| 98 | 96 | 89.4 | 84.54 |
| base_quest_false |	/	| /	| 95.43	| 97.71	| 91.36	| 100	| 100	| 97.61	| 80.12	| 96.67	| 98.71	| 100	| 99 | 98	| 97 | 96.27 |



问题：

这三个结果都是用full attention计算的，理应差距较小，但实际差距明显。尤其base_quest_false已经达到96的avg，但我看实际的生成结果，是存在一些错误答案的。请问有遇到过这样类似的情况吗，是哪里出了问题呢？


