'use client';

import { useState } from 'react';
import { Card, Tabs } from 'antd';
import { CodeBlock } from '@/app/components/CodeBlock';
import { Typography } from 'antd';
import styles from './page.module.css';
import React from 'react';

const { Title, Paragraph } = Typography;
const { TabPane } = Tabs;

export default function ControlFlowPage() {
  const [activeTab, setActiveTab] = useState('1');

  return (
    <div className={styles.container}>
      <Title level={1}>C++ 控制流程</Title>
      
      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="基础控制流" key="1">
          <Card title="条件语句" style={{ marginBottom: 16 }}>
            <Title level={3}>if 语句</Title>
            <Paragraph>最基本的条件判断：</Paragraph>
            <CodeBlock language="cpp">
              {`if (条件) {
    // 当条件为真时执行
}`}
            </CodeBlock>

            <Title level={3}>if-else 语句</Title>
            <Paragraph>提供两种选择：</Paragraph>
            <CodeBlock language="cpp">
              {`if (条件) {
    // 当条件为真时执行
} else {
    // 当条件为假时执行
}`}
            </CodeBlock>

            <Title level={3}>switch 语句</Title>
            <Paragraph>多分支选择结构：</Paragraph>
            <CodeBlock language="cpp">
              {`switch (表达式) {
    case 值1:
        // 当表达式等于值1时执行
        break;
    case 值2:
        // 当表达式等于值2时执行
        break;
    default:
        // 当表达式不等于任何case值时执行
}`}
            </CodeBlock>
          </Card>

          <Card title="循环语句" style={{ marginBottom: 16 }}>
            <Title level={3}>for 循环</Title>
            <Paragraph>适用于已知循环次数的情况：</Paragraph>
            <CodeBlock language="cpp">
              {`for (初始化; 条件; 更新) {
    // 循环体
}

// 范围for循环（C++11）
for (auto element : container) {
    // 对容器中的每个元素执行操作
}`}
            </CodeBlock>

            <Title level={3}>while 循环</Title>
            <Paragraph>当条件为真时重复执行：</Paragraph>
            <CodeBlock language="cpp">
              {`while (条件) {
    // 循环体
}`}
            </CodeBlock>

            <Title level={3}>do-while 循环</Title>
            <Paragraph>至少执行一次的循环：</Paragraph>
            <CodeBlock language="cpp">
              {`do {
    // 循环体
} while (条件);`}
            </CodeBlock>
          </Card>
        </TabPane>

        <TabPane tab="高级控制流" key="2">
          <Card title="现代C++特性" style={{ marginBottom: 16 }}>
            <Title level={3}>条件运算符（三元运算符）</Title>
            <CodeBlock language="cpp">
              {`结果 = (条件) ? 值1 : 值2;    // 如果条件为真返回值1，否则返回值2`}
            </CodeBlock>

            <Title level={3}>初始化语句的if和switch（C++17）</Title>
            <CodeBlock language="cpp">
              {`if (auto ptr = getValue(); ptr != nullptr) {
    // 使用ptr
}

switch (auto value = getValue(); value) {
    case 1: /* ... */ break;
    case 2: /* ... */ break;
}`}
            </CodeBlock>

            <Title level={3}>结构化绑定（C++17）</Title>
            <CodeBlock language="cpp">
              {`if (auto [iter, success] = map.insert(value); success) {
    // 插入成功的处理
}`}
            </CodeBlock>
          </Card>

          <Card title="异常处理" style={{ marginBottom: 16 }}>
            <Title level={3}>try-catch 块</Title>
            <CodeBlock language="cpp">
              {`try {
    // 可能抛出异常的代码
    throw exception();
} catch (异常类型1) {
    // 处理异常类型1
} catch (...) {
    // 处理所有其他类型的异常
}`}
            </CodeBlock>

            <Title level={3}>RAII（资源获取即初始化）</Title>
            <CodeBlock language="cpp">
              {`class ScopedLock {
    std::mutex& mutex;
public:
    ScopedLock(std::mutex& m) : mutex(m) { mutex.lock(); }
    ~ScopedLock() { mutex.unlock(); }
};`}
            </CodeBlock>
          </Card>
        </TabPane>

        <TabPane tab="示例代码" key="3">
          <Card title="完整示例" style={{ marginBottom: 16 }}>
            <CodeBlock language="cpp">
              {`#include <iostream>
using namespace std;

int main() {
    // 条件语句示例
    int score = 85;
    if (score >= 90) {
        cout << "优秀！" << endl;
    } else if (score >= 80) {
        cout << "良好！" << endl;
    } else if (score >= 60) {
        cout << "及格" << endl;
    } else {
        cout << "需要继续努力" << endl;
    }

    // 循环示例
    cout << "计数到5：" << endl;
    for (int i = 1; i <= 5; i++) {
        cout << i << " ";
    }
    cout << endl;

    return 0;
}`}
            </CodeBlock>
          </Card>
        </TabPane>
      </Tabs>
    </div>
  );
} 