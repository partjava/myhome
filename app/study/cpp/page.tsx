'use client';

import { Card, Progress, Button } from 'antd';
import { BookOutlined, RocketOutlined, TrophyOutlined } from '@ant-design/icons';
import Link from 'next/link';

const learningPath = [
  {
    title: '第一阶段：C++基础入门',
    items: [
      { name: '开发环境配置', href: '/study/cpp/setup', completed: false },
      { name: '基础语法', href: '/study/cpp/syntax', completed: false },
      { name: '变量和数据类型', href: '/study/cpp/variables', completed: false },
      { name: '运算符', href: '/study/cpp/operators', completed: false },
      { name: '控制流程', href: '/study/cpp/control', completed: false }
    ]
  },
  {
    title: '第二阶段：函数与数据结构',
    items: [
      { name: '函数', href: '/study/cpp/functions', completed: false },
      { name: '数组和字符串', href: '/study/cpp/arrays', completed: false },
      { name: '指针', href: '/study/cpp/pointers', completed: false },
      { name: '引用', href: '/study/cpp/references', completed: false }
    ]
  },
  {
    title: '第三阶段：面向对象编程',
    items: [
      { name: '结构体和类', href: '/study/cpp/structs', completed: false },
      { name: '面向对象编程', href: '/study/cpp/oop', completed: false }
    ]
  },
  {
    title: '第四阶段：高级特性',
    items: [
      { name: '模板编程', href: '/study/cpp/templates', completed: false },
      { name: 'STL标准库', href: '/study/cpp/stl', completed: false },
      { name: '文件操作', href: '/study/cpp/file-io', completed: false },
      { name: '异常处理', href: '/study/cpp/exceptions', completed: false },
      { name: '智能指针', href: '/study/cpp/smart-pointers', completed: false }
    ]
  },
  {
    title: '第五阶段：实战应用',
    items: [
      { name: '多线程编程', href: '/study/cpp/multithreading', completed: false },
      { name: '网络编程', href: '/study/cpp/networking', completed: false },
      { name: '项目实战', href: '/study/cpp/projects', completed: false }
    ]
  }
];

export default function CppLearningPage() {
  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* 课程概述 */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">C++ 编程学习之路</h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            从基础到进阶的完整学习路径，助你掌握C++编程，成为专业的开发者
          </p>
        </div>

        {/* 学习进度 */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          <Card className="text-center">
            <BookOutlined className="text-4xl text-blue-500 mb-4" />
            <h3 className="text-lg font-semibold mb-2">课程总数</h3>
            <p className="text-3xl font-bold text-gray-700">17</p>
          </Card>
          <Card className="text-center">
            <RocketOutlined className="text-4xl text-green-500 mb-4" />
            <h3 className="text-lg font-semibold mb-2">当前进度</h3>
            <Progress percent={0} />
          </Card>
          <Card className="text-center">
            <TrophyOutlined className="text-4xl text-yellow-500 mb-4" />
            <h3 className="text-lg font-semibold mb-2">学习时长</h3>
            <p className="text-3xl font-bold text-gray-700">0小时</p>
          </Card>
        </div>

        {/* 学习路径 */}
        <div className="space-y-8">
          {learningPath.map((phase, phaseIndex) => (
            <Card key={phaseIndex} title={phase.title} className="shadow-md">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {phase.items.map((item, itemIndex) => (
                  <Link key={itemIndex} href={item.href}>
                    <Card
                      hoverable
                      className={`text-center ${
                        item.completed ? 'bg-green-50' : 'bg-white'
                      }`}
                    >
                      <h4 className="text-lg font-medium mb-2">{item.name}</h4>
                      <div className="flex justify-center">
                        <Button type="primary">
                          {item.completed ? '复习' : '开始学习'}
                        </Button>
                      </div>
                    </Card>
                  </Link>
                ))}
              </div>
            </Card>
          ))}
        </div>

        {/* 学习建议 */}
        <Card className="mt-8">
          <h3 className="text-xl font-bold mb-4">学习建议</h3>
          <ul className="list-disc pl-6 space-y-2">
            <li>按照推荐的学习路径循序渐进，打好基础</li>
            <li>每个主题都配有理论讲解和实践练习，建议两者结合</li>
            <li>遇到不懂的概念可以随时回顾之前的内容</li>
            <li>动手实践很重要，每个主题都要完成配套的练习</li>
            <li>定期复习和总结，巩固所学知识</li>
          </ul>
        </Card>
      </div>
    </div>
  );
} 