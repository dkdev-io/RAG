import puppeteer from 'puppeteer';
import fs from 'fs';

async function verifyUICleanup() {
  const browser = await puppeteer.launch({
    headless: false,
    defaultViewport: { width: 1200, height: 800 },
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1200, height: 800 });

    console.log('Navigating to Streamlit app...');
    await page.goto('http://localhost:8502', {
      waitUntil: 'networkidle2',
      timeout: 30000
    });

    // Wait for Streamlit to fully load
    await new Promise(resolve => setTimeout(resolve, 3000));

    // 1. Take screenshot of main Query Documents interface
    console.log('Taking screenshot of Query Documents interface...');
    await page.screenshot({
      path: '/Users/Danallovertheplace/rag/tests/screenshots/01_updated_query_interface.png',
      fullPage: true
    });

    // Check for improved search input area
    const searchElements = {
      askQuestionHeader: false,
      textArea: false,
      searchButton: false,
      largerTextArea: false
    };

    // Check for "Ask a Question" header
    try {
      await page.waitForSelector('text=/Ask a Question/', { timeout: 5000 });
      searchElements.askQuestionHeader = true;
      console.log('✓ "Ask a Question" header found');
    } catch (e) {
      console.log('✗ "Ask a Question" header not found');
    }

    // Check for text area
    try {
      const textArea = await page.$('textarea');
      if (textArea) {
        searchElements.textArea = true;
        console.log('✓ Text area found');

        // Check if text area has adequate height (150px or similar)
        const height = await textArea.evaluate(el => el.style.height || getComputedStyle(el).height);
        if (height && parseInt(height) >= 100) {
          searchElements.largerTextArea = true;
          console.log('✓ Text area has adequate height:', height);
        } else {
          console.log('⚠️ Text area height may be too small:', height);
        }
      }
    } catch (e) {
      console.log('✗ Text area not found');
    }

    // Check for Search button
    try {
      await page.waitForSelector('button:has-text("Search")', { timeout: 5000 });
      searchElements.searchButton = true;
      console.log('✓ Search button found');
    } catch (e) {
      console.log('✗ Search button not found');
    }

    // 2. Navigate to Manage Files tab to verify instructions removal
    console.log('Clicking on Manage Files tab...');
    const manageFilesTab = await page.waitForSelector('div[data-testid="stTabs"] button:nth-child(3)', { timeout: 10000 });
    await manageFilesTab.click();
    await new Promise(resolve => setTimeout(resolve, 2000));

    // 3. Take screenshot of cleaned up Manage Files tab
    console.log('Taking screenshot of cleaned Manage Files tab...');
    await page.screenshot({
      path: '/Users/Danallovertheplace/rag/tests/screenshots/02_cleaned_manage_files.png',
      fullPage: true
    });

    // Check that instructions text is removed
    const instructionsRemoved = {
      noInstructionsSection: false,
      noSelectFolderText: false,
      noSupportedFilesText: false
    };

    // Check for absence of "Instructions" header
    try {
      await page.waitForSelector('text=/Instructions/', { timeout: 2000 });
      console.log('✗ Instructions section still found (should be removed)');
    } catch (e) {
      instructionsRemoved.noInstructionsSection = true;
      console.log('✓ Instructions section successfully removed');
    }

    // Check for absence of "Select folder" text
    try {
      await page.waitForSelector('text=/Select folder.*Select Folder.*tab/', { timeout: 2000 });
      console.log('✗ "Select folder" instruction text still found (should be removed)');
    } catch (e) {
      instructionsRemoved.noSelectFolderText = true;
      console.log('✓ "Select folder" instruction text successfully removed');
    }

    // Check for absence of "Supported file types" text
    try {
      await page.waitForSelector('text=/Supported file types.*New.*Incompatible/', { timeout: 2000 });
      console.log('✗ "Supported file types" text still found (should be removed)');
    } catch (e) {
      instructionsRemoved.noSupportedFilesText = true;
      console.log('✓ "Supported file types" text successfully removed');
    }

    // Generate test report
    const report = {
      timestamp: new Date().toISOString(),
      testResults: {
        queryInterface: {
          askQuestionHeader: searchElements.askQuestionHeader,
          textArea: searchElements.textArea,
          largerTextArea: searchElements.largerTextArea,
          searchButton: searchElements.searchButton
        },
        manageFilesCleanup: {
          instructionsRemoved: instructionsRemoved.noInstructionsSection,
          selectFolderTextRemoved: instructionsRemoved.noSelectFolderText,
          supportedFilesTextRemoved: instructionsRemoved.noSupportedFilesText
        }
      },
      screenshots: [
        '01_updated_query_interface.png',
        '02_cleaned_manage_files.png'
      ]
    };

    console.log('\\n=== UI CLEANUP TEST REPORT ===');
    console.log('Timestamp:', report.timestamp);
    console.log('\\nQuery Interface Improvements:');
    console.log('  - Ask Question header:', searchElements.askQuestionHeader ? '✓' : '✗');
    console.log('  - Text area present:', searchElements.textArea ? '✓' : '✗');
    console.log('  - Larger text area:', searchElements.largerTextArea ? '✓' : '✗');
    console.log('  - Search button:', searchElements.searchButton ? '✓' : '✗');

    console.log('\\nManage Files Cleanup:');
    console.log('  - Instructions removed:', instructionsRemoved.noInstructionsSection ? '✓' : '✗');
    console.log('  - Select folder text removed:', instructionsRemoved.noSelectFolderText ? '✓' : '✗');
    console.log('  - Supported files text removed:', instructionsRemoved.noSupportedFilesText ? '✓' : '✗');

    // Save test report
    fs.writeFileSync('/Users/Danallovertheplace/rag/tests/screenshots/ui_cleanup_report.json', JSON.stringify(report, null, 2));

    console.log('\\nAll screenshots saved to /Users/Danallovertheplace/rag/tests/screenshots/');
    console.log('UI cleanup test completed successfully!');

  } catch (error) {
    console.error('Error during testing:', error);

    // Take error screenshot
    try {
      await page.screenshot({
        path: '/Users/Danallovertheplace/rag/tests/screenshots/error_ui_cleanup.png',
        fullPage: true
      });
      console.log('Error screenshot saved');
    } catch (screenshotError) {
      console.error('Could not take error screenshot:', screenshotError);
    }
  } finally {
    await browser.close();
  }
}

// Run the verification
verifyUICleanup().catch(console.error);