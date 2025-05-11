'use client';
export default function SearchPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold mb-6">智能搜索引擎</h1>
      <div className="bg-white rounded-lg shadow p-8">
        <h2 className="text-2xl font-bold mb-4">课程目录</h2>
        <ul className="mb-4 space-y-2 text-gray-700">
          <li>• 搜索引擎基础</li>
          <li>• 爬虫与数据采集</li>
          <li>• 索引构建</li>
          <li>• 查询处理</li>
          <li>• Elasticsearch示例</li>
          <li>• 高级搜索特性</li>
        </ul>
        <h2 className="text-2xl font-bold mb-4">搜索引擎基础</h2>
        <ul className="mb-4 space-y-2 text-gray-700">
          <li>• 爬虫与数据采集</li>
          <li>• 索引构建</li>
          <li>• 查询处理</li>
        </ul>
        <h2 className="text-2xl font-bold mb-4">Elasticsearch示例</h2>
        <pre className="bg-gray-100 p-4 rounded-lg overflow-x-auto mb-6">
{`# 创建索引
PUT /my_index
{
  "mappings": {
    "properties": {
      "title": { "type": "text" },
      "content": { "type": "text" }
    }
  }
}

# 搜索文档
GET /my_index/_search
{
  "query": {
    "match": {
      "content": "搜索关键词"
    }
  }
}`}
        </pre>
        <h2 className="text-2xl font-bold mb-4">高级搜索特性</h2>
        <ul className="mb-4 space-y-2 text-gray-700">
          <li>• 模糊匹配</li>
          <li>• 同义词处理</li>
          <li>• 地理位置搜索</li>
        </ul>
      </div>
      <div className="mt-10 flex justify-between">
        <a href="/study/se/bigdata/projects" className="px-4 py-2 text-blue-600 hover:text-blue-800">
          ← 实战案例与项目
        </a>
        <a href="/study/se/modeling" className="px-4 py-2 text-blue-600 hover:text-blue-800">
          软件建模与设计 →
        </a>
      </div>
    </div>
  );
} 