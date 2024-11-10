目标

测试vllm_quest类，vllm类的运行结果。

其中，vllm_quest类可以选择启用或禁用quest方法，禁用quest方法时相当于一个重新用Pytorch实现了后台attention计算的vllm离线调用。

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

